
import os, re, math, json, time, unicodedata, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, mean_absolute_error
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

APP = "⚽ Football AI Coach V7"
ROOT = Path(".")
DATA_DIR = ROOT / "data_cache"
MODEL_DIR = ROOT / "model_store"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FIXTURES_URL = "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/resolve/main/fixtures.parquet"
STATS_URL = "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/resolve/main/match_stats.parquet"
ODDS_URL = "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/resolve/main/odds.parquet"
TEAMS_URL = "https://huggingface.co/datasets/eatpizzanot/soccer-dataset/resolve/main/teams.parquet"
MODEL_FILE = MODEL_DIR / "v7_bundle.joblib"
STATE_FILE = MODEL_DIR / "v7_state.json"
DATA_FILE = DATA_DIR / "matches.parquet"
STATS_FILE = DATA_DIR / "stats.parquet"
ODDS_FILE = DATA_DIR / "odds.parquet"
TEAMS_FILE = DATA_DIR / "teams.parquet"

ALIASES = {
    "psg":"Paris Saint-Germain", "paris saint germain":"Paris Saint-Germain",
    "paris sg":"Paris Saint-Germain", "man utd":"Manchester United",
    "man united":"Manchester United", "manchester utd":"Manchester United",
    "inter":"Inter Milan", "internazionale":"Inter Milan",
    "sporting lisbon":"Sporting CP", "sporting cp":"Sporting CP",
    "atletico madrid":"Atletico Madrid", "ath madrid":"Atletico Madrid",
    "bayern":"Bayern Munich", "bayern munich":"Bayern Munich",
    "real madrid cf":"Real Madrid", "barca":"Barcelona", "fc barcelona":"Barcelona",
    "juventus fc":"Juventus", "napoli fc":"Napoli", "ac milan":"AC Milan",
}

def norm(s):
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()
    s = s.lower().replace("&"," and ")
    s = re.sub(r"[^a-z0-9]+"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def canonical_input(s):
    n = norm(s)
    return norm(ALIASES.get(n, s))

@st.cache_data(ttl=86400, show_spinner=False)
def download_file(url, path):
    if path.exists() and path.stat().st_size > 1000:
        return str(path)
    r = requests.get(url, timeout=60, headers={"User-Agent":"FootballAICoach/7"})
    r.raise_for_status()
    path.write_bytes(r.content)
    return str(path)

def load_raw():
    download_file(FIXTURES_URL, DATA_FILE)
    download_file(TEAMS_URL, TEAMS_FILE)
    # match_stats/odds are optional; the model can operate from fixtures alone.
    try: download_file(STATS_URL, STATS_FILE)
    except Exception: pass
    try: download_file(ODDS_URL, ODDS_FILE)
    except Exception: pass
    fx = pd.read_parquet(DATA_FILE)
    teams = pd.read_parquet(TEAMS_FILE)
    if STATS_FILE.exists():
        stt = pd.read_parquet(STATS_FILE)
        keep = [c for c in stt.columns if c in [
            "fixture_id","home_goals_ht","away_goals_ht","home_xg","away_xg",
            "home_xg_ht","away_xg_ht"
        ]]
        if "fixture_id" in keep:
            fx = fx.merge(stt[keep].drop_duplicates("fixture_id"), left_on="id", right_on="fixture_id", how="left")
    if ODDS_FILE.exists():
        od = pd.read_parquet(ODDS_FILE)
        keep = [c for c in od.columns if c in ["fixture_id","home_win","draw","away_win","bookmaker","source","known_at"]]
        if "fixture_id" in keep:
            od = od[keep].copy()
            od = od.sort_values("known_at").drop_duplicates("fixture_id", keep="last") if "known_at" in od else od.drop_duplicates("fixture_id")
            fx = fx.merge(od, left_on="id", right_on="fixture_id", how="left")
    fx["date_utc"] = pd.to_datetime(fx["date_utc"], errors="coerce", utc=True)
    fx = fx.sort_values("date_utc").reset_index(drop=True)
    fx = fx[fx["home_team_id"].notna() & fx["away_team_id"].notna()].copy()
    fx["home_team_id"] = fx["home_team_id"].astype(int)
    fx["away_team_id"] = fx["away_team_id"].astype(int)
    team_map = dict(zip(teams["id"].astype(int), teams["name"].astype(str)))
    fx["home_name"] = fx["home_team_id"].map(team_map).fillna("Unknown")
    fx["away_name"] = fx["away_team_id"].map(team_map).fillna("Unknown")
    fx["played"] = fx["goals_home"].notna() & fx["goals_away"].notna()
    return fx, teams

def resolve_team(raw, teams):
    q = canonical_input(raw)
    exact = teams[teams["name"].map(norm) == q]
    if len(exact) == 1:
        r = exact.iloc[0]
        return int(r["id"]), str(r["name"]), "exact"
    # alias/canonical exact
    aliases = teams.copy()
    aliases["_n"] = aliases["name"].map(norm)
    target = norm(ALIASES.get(q, q))
    exact = aliases[aliases["_n"] == target]
    if len(exact) == 1:
        r = exact.iloc[0]
        return int(r["id"]), str(r["name"]), "alias"
    # safe token similarity, never substring-only
    toks = set(target.split())
    scores=[]
    for _,r in aliases.iterrows():
        rt=set(str(r["_n"]).split())
        if not rt: continue
        j=len(toks & rt)/max(1,len(toks|rt))
        if j >= 0.75:
            scores.append((j,int(r["id"]),str(r["name"])))
    scores.sort(reverse=True)
    if len(scores)==1 or (scores and (len(scores)==1 or scores[0][0]-scores[1][0] >= .12)):
        return scores[0][1], scores[0][2], "safe-fuzzy"
    return None, None, "ambiguous"

def init_team_state():
    return {}

def team_record(state, tid):
    return state.setdefault(int(tid), {
        "all":[], "home":[], "away":[], "elo":1500.0, "last_date":None
    })

def weighted_avg(vals, default=0.0):
    if not vals: return default
    vals=vals[-15:]
    w=np.exp(np.linspace(-1.5,0,len(vals)))
    return float(np.average(vals,weights=w))

def features_for(state, h, a, date):
    rh,ra=team_record(state,h),team_record(state,a)
    def pts(seq):
        return weighted_avg([x["pts"] for x in seq], 1.0)
    def gf(seq): return weighted_avg([x["gf"] for x in seq], 1.0)
    def ga(seq): return weighted_avg([x["ga"] for x in seq], 1.0)
    hd = rh["elo"]-ra["elo"]
    def rest(r):
        if r["last_date"] is None: return 30.0
        return max(0.0,(date-r["last_date"]).total_seconds()/86400)
    hh = rh["home"][-15:]; aa=ra["away"][-15:]
    h2h = []  # populated separately in training via global state if desired
    vals = [
        rh["elo"], ra["elo"], hd,
        pts(rh["all"][-5:]), pts(ra["all"][-5:]),
        pts(rh["all"][-10:]), pts(ra["all"][-10:]),
        gf(rh["all"][-5:]), ga(rh["all"][-5:]),
        gf(ra["all"][-5:]), ga(ra["all"][-5:]),
        pts(hh[-5:]), pts(aa[-5:]),
        gf(hh[-5:]), ga(hh[-5:]), gf(aa[-5:]), ga(aa[-5:]),
        rest(rh), rest(ra)
    ]
    return np.array(vals,dtype=float)

FEATURE_NAMES = [
    "home_elo","away_elo","elo_diff","home_form5","away_form5","home_form10","away_form10",
    "home_gf5","home_ga5","away_gf5","away_ga5","home_home_form5","away_away_form5",
    "home_home_gf5","home_home_ga5","away_away_gf5","away_away_ga5","home_rest","away_rest"
]

def update_state(state,h,a,date,hg,ag):
    rh,ra=team_record(state,h),team_record(state,a)
    exp_h=1/(1+10**((ra["elo"]-rh["elo"]-55)/400))
    result_h=1 if hg>ag else 0.5 if hg==ag else 0
    margin=max(1,abs(hg-ag))
    k=20*(1+math.log1p(margin))
    rh["elo"] += k*(result_h-exp_h)
    ra["elo"] += k*((1-result_h)-(1-exp_h))
    hp=3 if hg>ag else 1 if hg==ag else 0
    ap=3 if ag>hg else 1 if hg==ag else 0
    rh["all"].append({"gf":hg,"ga":ag,"pts":hp}); rh["home"].append({"gf":hg,"ga":ag,"pts":hp})
    ra["all"].append({"gf":ag,"ga":hg,"pts":ap}); ra["away"].append({"gf":ag,"ga":hg,"pts":ap})
    rh["all"]=rh["all"][-30:]; ra["all"]=ra["all"][-30:]
    rh["home"]=rh["home"][-30:]; ra["away"]=ra["away"][-30:]
    rh["last_date"]=date; ra["last_date"]=date

@st.cache_resource(show_spinner=True)
def train_or_load(fx):
    if MODEL_FILE.exists() and STATE_FILE.exists():
        try:
            bundle=joblib.load(MODEL_FILE)
            return bundle
        except Exception:
            pass
    played=fx[fx["played"]].copy()
    # Need HT data for FH/SH. Rows without HT are excluded from those targets.
    if "home_goals_ht" not in played.columns:
        played["home_goals_ht"]=np.nan; played["away_goals_ht"]=np.nan
    played["sh_home"]=played["goals_home"]-played["home_goals_ht"]
    played["sh_away"]=played["goals_away"]-played["away_goals_ht"]
    # Build chronological, leakage-safe features.
    state=init_team_state()
    X=[]; yft=[]; yfh=[]; ysh=[]; dates=[]
    for r in played.itertuples(index=False):
        d=r.date_utc
        if pd.isna(d): continue
        f=features_for(state,int(r.home_team_id),int(r.away_team_id),d)
        if not np.all(np.isfinite(f)): f=np.nan_to_num(f,nan=0.0,posinf=0.0,neginf=0.0)
        X.append(f); dates.append(d)
        hg,ag=int(r.goals_home),int(r.goals_away)
        yft.append(0 if hg>ag else 1 if hg==ag else 2)
        if pd.notna(r.home_goals_ht) and pd.notna(r.away_goals_ht):
            yfh.append((int(r.home_goals_ht),int(r.away_goals_ht)))
            ysh.append((int(r.sh_home),int(r.sh_away)))
        else:
            yfh.append((np.nan,np.nan)); ysh.append((np.nan,np.nan))
        update_state(state,int(r.home_team_id),int(r.away_team_id),d,hg,ag)
    X=np.asarray(X); yft=np.asarray(yft); dates=np.asarray(dates)
    n=len(X); cut=int(n*.82)
    # Cap training to keep Streamlit memory/time reasonable while preserving chronology.
    max_train=260000
    train_start=max(0,cut-max_train)
    tr=np.arange(train_start,cut); te=np.arange(cut,n)
    ft=HistGradientBoostingClassifier(max_iter=220,max_depth=7,learning_rate=.055,l2_regularization=1.5,random_state=42)
    ft.fit(X[tr],yft[tr])
    # FH/SH regressors use only rows with verified HT data.
    fh=np.asarray(yfh,dtype=float); sh=np.asarray(ysh,dtype=float)
    fhmask=np.isfinite(fh).all(axis=1); shmask=np.isfinite(sh).all(axis=1)
    ftr=np.intersect1d(tr,np.where(fhmask)[0]); fte=np.intersect1d(te,np.where(fhmask)[0])
    strr=np.intersect1d(tr,np.where(shmask)[0]); ste=np.intersect1d(te,np.where(shmask)[0])
    fh_h=HistGradientBoostingRegressor(max_iter=180,max_depth=6,learning_rate=.06,l2_regularization=1.0,random_state=1)
    fh_a=HistGradientBoostingRegressor(max_iter=180,max_depth=6,learning_rate=.06,l2_regularization=1.0,random_state=2)
    sh_h=HistGradientBoostingRegressor(max_iter=180,max_depth=6,learning_rate=.06,l2_regularization=1.0,random_state=3)
    sh_a=HistGradientBoostingRegressor(max_iter=180,max_depth=6,learning_rate=.06,l2_regularization=1.0,random_state=4)
    if len(ftr)>=500:
        fh_h.fit(X[ftr],fh[ftr,0]); fh_a.fit(X[ftr],fh[ftr,1])
    if len(strr)>=500:
        sh_h.fit(X[strr],sh[strr,0]); sh_a.fit(X[strr],sh[strr,1])
    p=ft.predict_proba(X[te])
    metrics={
        "training_rows":int(len(tr)),"test_rows":int(len(te)),
        "accuracy":float(accuracy_score(yft[te],np.argmax(p,axis=1))),
        "log_loss":float(log_loss(yft[te],p,labels=[0,1,2])),
        "brier":float(np.mean(np.sum((p-np.eye(3)[yft[te]])**2,axis=1))),
        "fh_training_rows":int(len(ftr)),"sh_training_rows":int(len(strr)),
        "trained_at":datetime.now(timezone.utc).isoformat(),
    }
    bundle={"ft":ft,"fh_h":fh_h,"fh_a":fh_a,"sh_h":sh_h,"sh_a":sh_a,
            "feature_names":FEATURE_NAMES,"metrics":metrics,"state":state}
    joblib.dump(bundle,MODEL_FILE,compress=3)
    STATE_FILE.write_text(json.dumps(metrics))
    return bundle

def score_matrix(lh,la,maxg=8):
    ps=np.outer(poisson.pmf(np.arange(maxg+1),max(1e-6,lh)),
                poisson.pmf(np.arange(maxg+1),max(1e-6,la)))
    return ps/ps.sum()

def one_x_two(mat):
    h=np.tril(mat,-1).sum() # row home goals > away? Need correct:
    h=sum(mat[i,j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i>j)
    d=sum(mat[i,j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i==j)
    a=sum(mat[i,j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i<j)
    return np.array([h,d,a])

def market_book(mat, fh_mat=None, sh_mat=None):
    def P(fn):
        return float(sum(mat[i,j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if fn(i,j)))
    out={
        "1":P(lambda i,j:i>j),"X":P(lambda i,j:i==j),"2":P(lambda i,j:i<j),
        "1X":P(lambda i,j:i>=j),"X2":P(lambda i,j:i<=j),"12":P(lambda i,j:i!=j),
        "Over 0.5":P(lambda i,j:i+j>0.5),"Over 1.5":P(lambda i,j:i+j>1.5),
        "Over 2.5":P(lambda i,j:i+j>2.5),"Over 3.5":P(lambda i,j:i+j>3.5),
        "Over 4.5":P(lambda i,j:i+j>4.5),
        "Under 1.5":P(lambda i,j:i+j<1.5),"Under 2.5":P(lambda i,j:i+j<2.5),
        "Under 3.5":P(lambda i,j:i+j<3.5),"Under 4.5":P(lambda i,j:i+j<4.5),
        "BTTS Yes":P(lambda i,j:i>=1 and j>=1),"BTTS No":P(lambda i,j:i==0 or j==0),
        "Home O0.5":P(lambda i,j:i>=1),"Home O1.5":P(lambda i,j:i>=2),"Home O2.5":P(lambda i,j:i>=3),
        "Away O0.5":P(lambda i,j:j>=1),"Away O1.5":P(lambda i,j:j>=2),"Away O2.5":P(lambda i,j:j>=3),
        "Home CS":P(lambda i,j:j==0),"Away CS":P(lambda i,j:i==0),
        "Home Win to Nil":P(lambda i,j:i>j and j==0),"Away Win to Nil":P(lambda i,j:j>i and i==0)
    }
    return {k:max(0,min(1,v)) for k,v in out.items()}

def compatible_ft(fh, sh):
    # Convolution of FH and SH score matrices.
    out=np.zeros((9,9))
    for i in range(fh.shape[0]):
        for j in range(fh.shape[1]):
            for k in range(sh.shape[0]):
                for l in range(sh.shape[1]):
                    if i+k<9 and j+l<9:
                        out[i+k,j+l]+=fh[i,j]*sh[k,l]
    return out/out.sum()

def best_scores(mat, n=10):
    vals=[]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            vals.append((float(mat[i,j]),i,j))
    return sorted(vals,reverse=True)[:n]

def current_team_stats(fx,tid, before_date):
    x=fx[(fx["played"]) & (fx["date_utc"]<before_date) & ((fx.home_team_id==tid)|(fx.away_team_id==tid))].tail(30)
    if x.empty: return {}
    return {"matches":len(x)}

def live_odds_search(home,away,date_text):
    # No bookmaker odds are invented. Public web search is best-effort only.
    # Search engines often block automation; failure is reported as unavailable.
    try:
        q=f"{home} vs {away} odds {date_text}".replace(" ","+")
        url="https://www.google.com/search?q="+q
        r=requests.get(url,timeout=8,headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code!=200: return None
        text=r.text
        # Conservative parser: only accept explicit decimal pairs in snippets.
        vals=re.findall(r'(?<![\d.])([1-9]\d?(?:\.\d{1,2})?)(?![\d.])',text)
        nums=[float(v) for v in vals if 1.01<=float(v)<=50]
        if len(nums)>=3:
            return {"home":nums[0],"draw":nums[1],"away":nums[2],"source":"public web search (unverified)"}
    except Exception:
        pass
    return None

st.set_page_config(page_title=APP,layout="wide")
st.title(APP)
st.caption("Real ML • FH/SH/FT • Poisson • Elo/Form • persistent model • no football API")

with st.sidebar:
    st.subheader("MATCH")
    home_raw=st.text_input("HOME TEAM","Paris Saint-Germain")
    away_raw=st.text_input("AWAY TEAM","Lens")
    date_raw=st.text_input("MATCH DATE (optional)","")
    st.divider()
    refresh=st.button("Refresh public dataset")
    if refresh:
        for p in [DATA_FILE,STATS_FILE,ODDS_FILE,TEAMS_FILE,MODEL_FILE,STATE_FILE]:
            try:p.unlink()
            except:pass
        st.cache_data.clear(); st.cache_resource.clear()
        st.rerun()

try:
    fx,teams=load_raw()
except Exception as e:
    st.error("DATA LOAD FAILED")
    st.exception(e); st.stop()

bundle=train_or_load(fx)
m=bundle["metrics"]
st.success(f"MODEL READY • persisted • historical fixtures loaded: {len(fx):,}")
c1,c2,c3,c4=st.columns(4)
c1.metric("Training rows",f"{m['training_rows']:,}")
c2.metric("Test rows",f"{m['test_rows']:,}")
c3.metric("Accuracy",f"{m['accuracy']*100:.2f}%")
c4.metric("Log Loss",f"{m['log_loss']:.4f}")

hid,hname,hmode=resolve_team(home_raw,teams)
aid,aname,amode=resolve_team(away_raw,teams)
if not hid or not aid or hid==aid:
    st.error("TEAMS NOT SAFELY RESOLVED")
    st.write("Home:",hmode,"• Away:",amode)
    st.stop()
st.success(f"TEAMS CONFIRMED • {hname} vs {aname}")

# Determine a prediction date; use supplied date or next relevant calendar date.
if date_raw:
    try: target_date=pd.Timestamp(date_raw,tz="UTC")
    except: target_date=pd.Timestamp.now(tz="UTC")
else:
    target_date=pd.Timestamp.now(tz="UTC")

# Reconstruct current state by replaying all played matches before target.
state=init_team_state()
hist=fx[(fx.played)&(fx.date_utc<target_date)].sort_values("date_utc")
for r in hist.itertuples(index=False):
    update_state(state,int(r.home_team_id),int(r.away_team_id),r.date_utc,int(r.goals_home),int(r.goals_away))
feat=features_for(state,hid,aid,target_date).reshape(1,-1)
feat=np.nan_to_num(feat,nan=0,posinf=0,neginf=0)

p_ft_ml=bundle["ft"].predict_proba(feat)[0]
# Model-derived expected goals; regressors are independently trained on real HT/SH targets.
fh_h=float(max(0,bundle["fh_h"].predict(feat)[0]))
fh_a=float(max(0,bundle["fh_a"].predict(feat)[0]))
sh_h=float(max(0,bundle["sh_h"].predict(feat)[0]))
sh_a=float(max(0,bundle["sh_a"].predict(feat)[0]))
ft_h,ft_a=fh_h+sh_h,fh_a+sh_a

fh_mat=score_matrix(fh_h,fh_a)
sh_mat=score_matrix(sh_h,sh_a)
ft_compat=compatible_ft(fh_mat,sh_mat)
p_ft_poi=one_x_two(ft_compat)
p_ft=(0.60*p_ft_ml+0.40*p_ft_poi); p_ft=p_ft/p_ft.sum()

st.header("FIRST HALF")
st.write(f"Expected goals: **{fh_h:.2f} — {fh_a:.2f}**")
fh1=one_x_two(fh_mat)
fc1,fcd,fca=st.columns(3); fc1.metric("Home",f"{fh1[0]*100:.1f}%"); fcd.metric("Draw",f"{fh1[1]*100:.1f}%"); fca.metric("Away",f"{fh1[2]*100:.1f}%")
fs=best_scores(fh_mat,1)[0]
st.write(f"Most likely FH score: **{fs[1]}–{fs[2]} ({fs[0]*100:.1f}%)**")

st.header("SECOND HALF")
st.write(f"Expected goals: **{sh_h:.2f} — {sh_a:.2f}**")
sh1=one_x_two(sh_mat)
sc1,scd,sca=st.columns(3); sc1.metric("Home",f"{sh1[0]*100:.1f}%"); scd.metric("Draw",f"{sh1[1]*100:.1f}%"); sca.metric("Away",f"{sh1[2]*100:.1f}%")
ss=best_scores(sh_mat,1)[0]
st.write(f"Most likely SH score: **{ss[1]}–{ss[2]} ({ss[0]*100:.1f}%)**")

st.header("FULL TIME")
st.write(f"Expected goals: **{ft_h:.2f} — {ft_a:.2f}**")
fc1,fcd,fca=st.columns(3); fc1.metric("Home",f"{p_ft[0]*100:.1f}%"); fcd.metric("Draw",f"{p_ft[1]*100:.1f}%"); fca.metric("Away",f"{p_ft[2]*100:.1f}%")
fs=best_scores(ft_compat,1)[0]
st.success(f"MOST LIKELY FINAL SCORE: {fs[1]}–{fs[2]} • {fs[0]*100:.1f}%")

st.header("TOP 10 SCORES")
rows=[{"Rank":i+1,"Score":f"{h}–{a}","Probability":f"{p*100:.2f}%","Fair Odds":f"{1/p:.2f}"} for i,(p,h,a) in enumerate(best_scores(ft_compat,10))]
st.dataframe(pd.DataFrame(rows),use_container_width=True)

st.header("MARKETS")
pm=market_book(ft_compat)
mk=pd.DataFrame([{"Market":k,"Probability":f"{v*100:.1f}%","Fair Odds":f"{1/v:.2f}"} for k,v in sorted(pm.items(),key=lambda z:z[1],reverse=True)])
st.dataframe(mk,use_container_width=True)

st.header("BEST BET")
# Rank diversified markets by model probability; no value claim without current bookmaker odds.
families={
"1":"1X2","X":"1X2","2":"1X2","1X":"Double Chance","X2":"Double Chance","12":"Double Chance",
"Over 0.5":"Goals","Over 1.5":"Goals","Over 2.5":"Goals","Over 3.5":"Goals","Over 4.5":"Goals",
"Under 1.5":"Goals","Under 2.5":"Goals","Under 3.5":"Goals","Under 4.5":"Goals",
"BTTS Yes":"BTTS","BTTS No":"BTTS","Home O0.5":"Team Goals","Home O1.5":"Team Goals","Home O2.5":"Team Goals",
"Away O0.5":"Team Goals","Away O1.5":"Team Goals","Away O2.5":"Team Goals",
"Home CS":"Clean Sheet","Away CS":"Clean Sheet","Home Win to Nil":"Win to Nil","Away Win to Nil":"Win to Nil"}
chosen=[]; used=set()
for name,p in sorted(pm.items(),key=lambda z:z[1],reverse=True):
    fam=families.get(name,name)
    if fam in used: continue
    used.add(fam); chosen.append((name,p))
    if len(chosen)>=3: break
best_rows=[]
for rank,(name,p) in enumerate(chosen,1):
    best_rows.append({"Rank":rank,"Market":name,"Model Probability":f"{p*100:.1f}%","Fair Odds":f"{1/p:.2f}","Bookmaker Odds":"NOT AVAILABLE","Value":"NOT CALCULABLE"})
st.dataframe(pd.DataFrame(best_rows),use_container_width=True)
st.info("BEST BET هنا هو أفضل اختيار حسب احتمال النموذج فقط. لا توجد قيمة مالية مؤكدة بدون سعر bookmaker حقيقي. لا يتم اختراع Odds.")

# Try a conservative public-web odds lookup only after model output.
if st.button("Try public-web odds lookup"):
    odds=live_odds_search(hname,aname,date_raw or "upcoming")
    if odds:
        st.warning("تم العثور على أرقام محتملة من بحث الويب، لكنها غير موثقة كمصدر bookmaker؛ لذلك لا تدخل في حساب Value.")
        st.json(odds)
    else:
        st.info("BOOKMAKER ODDS NOT FOUND FROM PUBLIC WEB")

st.header("MODEL HEALTH")
st.json({**m,"model_type":"HistGradientBoostingClassifier + 4 HistGradientBoostingRegressors","persistent_model":MODEL_FILE.exists(),
         "feature_count":len(FEATURE_NAMES),"continual_learning":"incremental result ingestion module not yet auto-connected to live web results"})

st.caption("المصدر الأولي للـbootstrap هو dataset عام CC-BY يحتوي على HT/FT وOdds. هذا الإصدار لا يستخدم football-data.org أو API-Football مباشرة. بيانات Odds التاريخية ليست Odds حية للمباراة الحالية.")
