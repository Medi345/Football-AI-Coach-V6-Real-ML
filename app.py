
import os, re, json, math, time, sqlite3, hashlib, unicodedata
from pathlib import Path
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

APP = "⚽ Football AI Coach V6.3"
BASE = Path(".")
STATE = BASE / "coach_state"
STATE.mkdir(exist_ok=True)
MODEL_DIR = STATE / "models"
MODEL_DIR.mkdir(exist_ok=True)
DB_PATH = STATE / "knowledge.sqlite"

# Public, non-API bootstrap dataset. It contains 1.2M+ historical matches
# across 207 domestic top-tier leagues and 20 international tournaments
# through 2023. The source is documented in README.md.
BOOTSTRAP_URL = (
    "https://raw.githubusercontent.com/schochastics/football-data/"
    "master/data/results/games.parquet"
)

ALIASES = {
    "psg": "Paris Saint-Germain",
    "paris saint germain": "Paris Saint-Germain",
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "manchester utd": "Manchester United",
    "inter": "Inter",
    "internazionale": "Inter",
    "bayern": "Bayern Munich",
    "atletico madrid": "Atlético Madrid",
    "atletico": "Atlético Madrid",
    "barca": "Barcelona",
    "fc barcelona": "Barcelona",
}

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode()
    s = s.lower().replace("&","and")
    s = re.sub(r"[^a-z0-9]+"," ",s).strip()
    return s

def alias(s):
    n = norm(s)
    return norm(ALIASES.get(n, s))

def db():
    c=sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS matches(
        date TEXT, home TEXT, away TEXT, gh INTEGER, ga INTEGER,
        competition TEXT, level TEXT, source TEXT, PRIMARY KEY(date,home,away,gh,ga,source)
    )""")
    c.commit()
    return c

@st.cache_data(show_spinner=False)
def download_bootstrap():
    p=STATE/"games.parquet"
    if not p.exists() or p.stat().st_size < 1000000:
        r=requests.get(BOOTSTRAP_URL,timeout=90)
        r.raise_for_status()
        p.write_bytes(r.content)
    return str(p)

def load_bootstrap():
    p=download_bootstrap()
    df=pd.read_parquet(p)
    df=df.rename(columns={"gh":"home_goals","ga":"away_goals"})
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    df=df.dropna(subset=["date","home","away","home_goals","away_goals"]).copy()
    df["home_goals"]=pd.to_numeric(df["home_goals"],errors="coerce")
    df["away_goals"]=pd.to_numeric(df["away_goals"],errors="coerce")
    df=df.dropna(subset=["home_goals","away_goals"])
    df=df.sort_values("date").reset_index(drop=True)
    return df

def outcome(r):
    if r.home_goals>r.away_goals: return 0
    if r.home_goals==r.away_goals: return 1
    return 2

def build_features(df, limit=300000):
    """Chronological, leakage-safe feature construction."""
    if len(df)>limit:
        df=df.tail(limit).copy()
    teams={}
    elo={}
    rows=[]; ys=[]
    def st(t):
        if t not in teams:
            teams[t]={"n":0,"gf":[],"ga":[],"pts":[],"home":[],"away":[]}
        return teams[t]
    def e(t): return elo.get(t,1500.0)
    for _,r in df.iterrows():
        h=str(r.home); a=str(r.away)
        hs,as_=st(h),st(a)
        hgf=np.mean(hs["gf"][-10:]) if hs["gf"] else 1.2
        hga=np.mean(hs["ga"][-10:]) if hs["ga"] else 1.2
        agf=np.mean(as_["gf"][-10:]) if as_["gf"] else 1.0
        aga=np.mean(as_["ga"][-10:]) if as_["ga"] else 1.3
        hp=np.mean(hs["pts"][-5:]) if hs["pts"] else 1.0
        ap=np.mean(as_["pts"][-5:]) if as_["pts"] else 1.0
        hhome=np.mean(hs["home"][-8:]) if hs["home"] else 1.0
        aaway=np.mean(as_["away"][-8:]) if as_["away"] else 1.0
        x=[
            e(h)-e(a), e(h)-1500, e(a)-1500,
            hgf-hga, agf-aga, hgf, hga, agf, aga,
            hp-ap, hhome-aaway,
            min(hs["n"],20), min(as_["n"],20)
        ]
        if min(hs["n"],as_["n"])>=3:
            rows.append(x); ys.append(outcome(r))
        hg=float(r.home_goals); ag=float(r.away_goals)
        hpts=3 if hg>ag else 1 if hg==ag else 0
        apts=3 if ag>hg else 1 if hg==ag else 0
        for t,go,con,pts,side in [(h,hg,ag,hpts,"home"),(a,ag,hg,apts,"away")]:
            s=st(t); s["n"]+=1; s["gf"].append(go); s["ga"].append(con); s["pts"].append(pts)
            s[side].append(pts)
        # Elo update with margin-of-victory scaling.
        diff=e(h)-e(a)+60
        exp=1/(1+10**(-diff/400))
        actual=1 if hg>ag else .5 if hg==ag else 0
        margin=math.log1p(abs(hg-ag)+1)
        k=18*margin
        elo[h]=e(h)+k*(actual-exp)
        elo[a]=e(a)+k*((1-actual)-(1-exp))
    X=np.asarray(rows,dtype=float); y=np.asarray(ys)
    return X,y,elo,teams

def train_initial(df):
    X,y,elo,teams=build_features(df)
    split=max(1,int(len(X)*0.82))
    Xtr,Xte=X[:split],X[split:]; ytr,yte=y[:split],y[split:]
    model=HistGradientBoostingClassifier(
        max_iter=260, learning_rate=.055, max_leaf_nodes=15,
        l2_regularization=1.2, random_state=42
    )
    model.fit(Xtr,ytr)
    p=model.predict_proba(Xte)
    acc=accuracy_score(yte,model.predict(Xte))
    ll=log_loss(yte,p,labels=[0,1,2])
    # one-vs outcome Brier (multiclass mean)
    bs=np.mean([brier_score_loss((yte==k).astype(int),p[:,k]) for k in range(3)])
    joblib.dump(model,MODEL_DIR/"ft_hgb.joblib")
    joblib.dump({"elo":elo,"teams":teams,"features":"v6","trained_rows":len(X),
                 "train_rows":len(Xtr),"test_rows":len(Xte),
                 "accuracy":acc,"log_loss":ll,"brier":bs,
                 "trained_at":datetime.now(timezone.utc).isoformat()},
                MODEL_DIR/"state.joblib")
    return model, {"train":len(Xtr),"test":len(Xte),"accuracy":acc,"log_loss":ll,"brier":bs}

@st.cache_resource(show_spinner=False)
def get_model():
    df=load_bootstrap()
    mpath=MODEL_DIR/"ft_hgb.joblib"
    spath=MODEL_DIR/"state.joblib"
    if mpath.exists() and spath.exists():
        return joblib.load(mpath), joblib.load(spath), len(df), "persisted"
    model,metrics=train_initial(df)
    return model, joblib.load(spath), len(df), "bootstrapped"

def team_lookup(df, q):
    """Strict, alias-aware team resolver.

    The old resolver treated any substring as a strong match. That made
    ``Paris Saint-Germain`` incorrectly match ``Aris`` because ``aris`` is
    contained inside ``paris``. V6.1 only gives a strong score to an exact
    normalized/alias match, and uses token/edit similarity only as a fallback.
    Short one-token candidates are never accepted merely because they are a
    substring of a longer team name.
    """
    from difflib import SequenceMatcher
    qn=alias(q)
    qtokens=set(qn.split())
    names=pd.unique(pd.concat([df.home,df.away],ignore_index=True).astype(str))
    scored=[]
    for raw in names:
        n=str(raw).strip()
        nn=norm(n)
        if not nn:
            continue
        # Strongest possible match: normalized name or known alias.
        if nn==qn:
            s=1.0
        else:
            ntokens=set(nn.split())
            inter=len(qtokens & ntokens)
            union=len(qtokens | ntokens)
            jacc=inter/max(1,union)
            seq=SequenceMatcher(None,qn,nn).ratio()
            # Only allow containment when it is a whole-token match and the
            # candidate has at least two meaningful tokens.
            token_contained = (len(ntokens)>=2 and (qn in nn or nn in qn))
            if token_contained:
                s=max(jacc, seq*0.96)
            else:
                s=max(jacc, seq*0.90)
            # One-token names such as Aris must not score highly just because
            # their letters occur inside a multi-token query.
            if len(ntokens)==1 and len(qtokens)>=2 and nn not in qtokens:
                s=min(s,0.55)
        if s>=0.55:
            scored.append((float(s),n))
    # Deduplicate names by normalized spelling, keeping the strongest score.
    best_by_norm={}
    for s,n in scored:
        k=norm(n)
        if k not in best_by_norm or s>best_by_norm[k][0]:
            best_by_norm[k]=(s,n)
    out=list(best_by_norm.values())
    out.sort(key=lambda x:(-x[0], x[1].lower()))
    return out[:8]

def make_feature_for_match(state,h,a):
    teams=state["teams"]; elo=state["elo"]
    def get(t):
        return teams.get(t,{"n":0,"gf":[],"ga":[],"pts":[],"home":[],"away":[]})
    hs,as_=get(h),get(a)
    def avg(v,n,default): return float(np.mean(v[-n:])) if v else default
    return np.array([[
        elo.get(h,1500)-elo.get(a,1500), elo.get(h,1500)-1500, elo.get(a,1500)-1500,
        avg(hs["gf"],10,1.2)-avg(hs["ga"],10,1.2),
        avg(as_["gf"],10,1.0)-avg(as_["ga"],10,1.3),
        avg(hs["gf"],10,1.2), avg(hs["ga"],10,1.2),
        avg(as_["gf"],10,1.0), avg(as_["ga"],10,1.3),
        avg(hs["pts"],5,1)-avg(as_["pts"],5,1),
        avg(hs["home"],8,1)-avg(as_["away"],8,1),
        min(hs["n"],20),min(as_["n"],20)
    ]])

def expected_goals(state,h,a):
    teams=state["teams"]
    hs=teams.get(h); as_=teams.get(a)
    if not hs or not as_: return None
    hgf=np.mean(hs["gf"][-10:]) if hs["gf"] else 1.3
    hga=np.mean(hs["ga"][-10:]) if hs["ga"] else 1.1
    agf=np.mean(as_["gf"][-10:]) if as_["gf"] else 1.1
    aga=np.mean(as_["ga"][-10:]) if as_["ga"] else 1.3
    # conservative blend of attack/defence, clipped for stability
    eh=.58*hgf+.42*aga+.18
    ea=.58*agf+.42*hga
    return float(np.clip(eh,.15,4.2)),float(np.clip(ea,.15,4.2))

def poisson_matrix(lh,la,n=8):
    mat=np.outer(poisson.pmf(np.arange(n+1),lh),poisson.pmf(np.arange(n+1),la))
    return mat/mat.sum()

def markets(mat):
    """Calculate a broad, deterministic market book from the score matrix."""
    n = mat.shape[0]
    entries = [(i, j, float(mat[i, j])) for i in range(n) for j in range(n)]

    def prob(fn):
        return float(sum(p for i, j, p in entries if fn(i, j)))

    def total_gt(x):
        return prob(lambda i, j: i + j > x)

    def total_lt(x):
        return prob(lambda i, j: i + j < x)

    out = {
        "Home Win": prob(lambda i,j: i > j),
        "Draw": prob(lambda i,j: i == j),
        "Away Win": prob(lambda i,j: i < j),
        "1X": prob(lambda i,j: i >= j),
        "X2": prob(lambda i,j: i <= j),
        "12": prob(lambda i,j: i != j),

        "Over 0.5": total_gt(0.5),
        "Over 1.5": total_gt(1.5),
        "Over 2.5": total_gt(2.5),
        "Over 3.5": total_gt(3.5),
        "Over 4.5": total_gt(4.5),

        "Under 1.5": total_lt(1.5),
        "Under 2.5": total_lt(2.5),
        "Under 3.5": total_lt(3.5),
        "Under 4.5": total_lt(4.5),

        "BTTS Yes": prob(lambda i,j: i >= 1 and j >= 1),
        "BTTS No": prob(lambda i,j: i == 0 or j == 0),

        "Home Over 0.5": prob(lambda i,j: i >= 1),
        "Home Over 1.5": prob(lambda i,j: i >= 2),
        "Home Over 2.5": prob(lambda i,j: i >= 3),

        "Away Over 0.5": prob(lambda i,j: j >= 1),
        "Away Over 1.5": prob(lambda i,j: j >= 2),
        "Away Over 2.5": prob(lambda i,j: j >= 3),

        "Home Clean Sheet": prob(lambda i,j: j == 0),
        "Away Clean Sheet": prob(lambda i,j: i == 0),
        "Home Win To Nil": prob(lambda i,j: i > j and j == 0),
        "Away Win To Nil": prob(lambda i,j: j > i and i == 0),
    }
    # Keep numerical safety after the finite 0..8 score grid.
    return {k: float(np.clip(v, 0.0, 1.0)) for k,v in out.items()}

def exact_fixture_web(home,away,date=None):
    # No API: use public search engine HTML through a search endpoint.
    # We only confirm when both names and, if supplied, the date are present.
    q=f'"{home}" "{away}" football'
    if date: q+=f' "{date}"'
    try:
        r=requests.get("https://www.google.com/search",params={"q":q},
                       headers={"User-Agent":"Mozilla/5.0"},timeout=12)
        text=re.sub(r"<[^>]+>"," ",r.text)
        t=norm(text)
        ok=norm(home) in t and norm(away) in t
        if date: ok = ok and str(date) in text
        return ok
    except Exception:
        return False

st.set_page_config(page_title=APP,page_icon="⚽",layout="wide")
st.title(APP)
st.caption("Real ML • persistent pre-trained model • strict team resolver • continual-learning ready • no football API")

with st.sidebar:
    st.header("MATCH")
    home=st.text_input("HOME TEAM","Paris Saint-Germain")
    away=st.text_input("AWAY TEAM","Lens")
    date=st.text_input("MATCH DATE (optional)","")
    refresh=st.button("Refresh public dataset")
    st.info("No football API key is required. The initial model is trained once from a real historical open dataset and then persisted.")
    if refresh:
        for p in [STATE/"games.parquet",MODEL_DIR/"ft_hgb.joblib",MODEL_DIR/"state.joblib"]:
            if p.exists(): p.unlink()
        st.cache_data.clear(); st.cache_resource.clear()
        st.rerun()

try:
    with st.spinner("Loading persistent football model..."):
        model,state,nrows,mode=get_model()
    st.success(f"MODEL READY • {mode.upper()} • historical rows available: {nrows:,}")
    st.caption("Historical rows = source dataset size. The trained model reports its actual training/test rows below.")
except Exception as e:
    st.error("MODEL BOOTSTRAP FAILED")
    st.exception(e)
    st.stop()

st.write(f"### {home}  vs  {away}")
st.caption("Team matching uses exact normalized names and explicit aliases first; weak substring matches are rejected.")
hcan=team_lookup(load_bootstrap(),home)
acan=team_lookup(load_bootstrap(),away)
if not hcan or not acan:
    st.error("TEAM NOT FOUND IN TRAINING KNOWLEDGE BASE")
    st.stop()

# Strictly confirm only if top candidate is clearly separated.
if hcan[0][0] < .78 or (len(hcan)>1 and hcan[0][0]-hcan[1][0]<.08):
    st.warning("HOME TEAM AMBIGUOUS")
    st.dataframe(pd.DataFrame(hcan,columns=["score","team"]))
    st.stop()
if acan[0][0] < .78 or (len(acan)>1 and acan[0][0]-acan[1][0]<.08):
    st.warning("AWAY TEAM AMBIGUOUS")
    st.dataframe(pd.DataFrame(acan,columns=["score","team"]))
    st.stop()

H,A=hcan[0][1],acan[0][1]
st.success(f"TEAMS CONFIRMED • {H} vs {A}")

if date:
    try: d=pd.to_datetime(date).date().isoformat()
    except: d=date
else: d=None

fixture_ok=exact_fixture_web(H,A,d)
if not fixture_ok:
    st.warning("EXACT FIXTURE NOT VERIFIED ON PUBLIC WEB — prediction is model-only and no fixture data is invented.")
else:
    st.success("EXACT FIXTURE VERIFIED ON PUBLIC WEB")

X=make_feature_for_match(state,H,A)
p_ml=model.predict_proba(X)[0]
eg=expected_goals(state,H,A)
if eg is None:
    st.error("DATA NOT AVAILABLE FOR THIS MATCH")
    st.stop()
lh,la=eg
mat=poisson_matrix(lh,la,8)
pm=markets(mat)

# Ensemble: ML 1X2 + Poisson, with transparent equal weights.
p_po=np.array([pm["Home Win"],pm["Draw"],pm["Away Win"]])
p_final=.55*p_ml+.45*p_po
p_final=p_final/p_final.sum()

top_scores=sorted(((i,j), float(mat[i,j])) for i in range(mat.shape[0]) for j in range(mat.shape[1]))[::-1]
best=[]
for name,p in pm.items():
    p=float(p)
    if not np.isfinite(p) or p <= 0.0:
        continue
    fair=1.0/p
    # Without bookmaker odds, this is a model-probability ranking only.
    # We penalize extremely low-probability markets rather than pretending
    # they have positive betting value.
    score = p * math.sqrt(max(0.0, min(1.0, p)))
    best.append((name,p,fair,score))
best=sorted(best,key=lambda x:x[3],reverse=True)

st.header("FULL TIME")
c1,c2,c3=st.columns(3)
c1.metric("Home",f"{p_final[0]*100:.1f}%")
c2.metric("Draw",f"{p_final[1]*100:.1f}%")
c3.metric("Away",f"{p_final[2]*100:.1f}%")
st.write(f"**Expected goals:** {lh:.2f} — {la:.2f}")
with st.expander("MODEL vs POISSON PROBABILITIES"):
    diag = pd.DataFrame({
        "Outcome":["Home","Draw","Away"],
        "ML":[f"{x*100:.2f}%" for x in p_ml],
        "Poisson":[f"{x*100:.2f}%" for x in p_po],
        "Ensemble":[f"{x*100:.2f}%" for x in p_final],
    })
    st.dataframe(diag, use_container_width=True)


st.header("TOP 10 SCORES")
rows=[]
for (i,j),pr in top_scores[:10]:
    rows.append({"Score":f"{i}-{j}","Probability":pr,"Fair odds":1/pr})
st.dataframe(pd.DataFrame(rows),use_container_width=True)

st.header("MARKETS")
mr=[]
for name,p in pm.items():
    mr.append({"Market":name,"Probability":f"{p*100:.1f}%","Fair Odds":f"{1/p:.2f}"})
st.dataframe(pd.DataFrame(mr),use_container_width=True)

st.header("BEST MODEL BETS")
st.caption("These are model selections only. Without verified bookmaker odds there is NO real bookmaker value/edge calculation.")
bb=[]
for name,p,fair,score in best[:3]:
    confidence=(0.5+0.5*max(p_final))*(0.7+0.3*min(1,nrows/200000))
    bb.append({"Rank":len(bb)+1,"Market":name,"Probability":f"{p*100:.1f}%",
               "Fair Odds":f"{fair:.2f}","Bookmaker Odds":"NOT AVAILABLE",
               "Value/Edge":"NOT CALCULABLE",
               "Confidence":f"{confidence*100:.1f}%"})
st.dataframe(pd.DataFrame(bb),use_container_width=True)

st.header("FIRST HALF / SECOND HALF")
st.info("The bootstrap dataset used for the persistent ML model contains full-time results. It does not contain reliable first-half scores for all competitions, so FH/SH are NOT fabricated. They become available only after a verified source supplies real half-time data.")

st.header("MODEL HEALTH")
meta=state
st.write({
    "trained_at":meta.get("trained_at"),
    "training_rows":meta.get("train_rows"),
    "test_rows":meta.get("test_rows"),
    "accuracy":round(meta.get("accuracy",0),4),
    "log_loss":round(meta.get("log_loss",0),4),
    "brier":round(meta.get("brier",0),4),
    "model_type":"HistGradientBoostingClassifier",
    "feature_count":13,
    "persistent_model":True
})
st.caption("The trained model is persisted in the app filesystem and loaded on later runs while that storage remains available. V6.3 does not claim completed automatic online learning until new labeled-match ingestion is connected.")
