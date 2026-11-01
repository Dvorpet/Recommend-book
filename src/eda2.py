import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
D=Path(sys.argv[1])
books=pd.read_csv(D/"Books.csv",dtype=str,keep_default_na=False,na_values=[""])
r=pd.read_csv(D/"Ratings.csv",dtype={"ISBN":str})
r["impl"]=r["Book-Rating"]==0
g=r.groupby("User-ID").agg(n=("ISBN","size"),impl=("impl","mean"))
top=g.sort_values("n",ascending=False).head(10)
print("top users implicit share:\n",top)
print("users >1000 ratings:",(g.n>1000).sum(),"share of ratings:",g.n[g.n>1000].sum()/len(r))
print("all-implicit users:",(g.impl==1).sum(),"/",len(g), " share of ratings by all-implicit users:", g.n[g.impl==1].sum()/len(r))
m=r[r.ISBN.isin(set(books.ISBN))]
gm=m.groupby("User-ID").size()
print("users with >=5 matched interactions:",(gm>=5).sum(), " >=3:",(gm>=3).sum(), " ==1:",(gm==1).sum(), "of",len(gm))
# html entities / mojibake
t=books["Book-Title"]
print("titles with &amp; etc:",t.str.contains(r"&amp;|&quot;|&#",regex=True).sum())
print("titles with mojibake (Ã):",t.str.contains("Ã",regex=False).sum())
a=books["Book-Author"].fillna("")
print("authors with mojibake:",a.str.contains("Ã",regex=False).sum())
# author name variants e.g. Stephen King
print(books[a.str.lower().str.contains("stephen king")]["Book-Author"].value_counts().head(8))
# publishers variants
print("publisher ISBN prefix not same:")
# wild animus
wa=r[r.ISBN=="0971880107"]["Book-Rating"].value_counts().sort_index()
print("Wild Animus dist:",wa.to_dict())
# explicit dist excluding 0 for the chart
# Bayesian avg top books vs count top books
mm=m.merge(books[["ISBN","Book-Title","Book-Author"]],on="ISBN")
e=mm[mm["Book-Rating"]>0]
C=e["Book-Rating"].mean()
agg=e.groupby(["ISBN","Book-Title","Book-Author"])["Book-Rating"].agg(["count","mean"]).reset_index()
mq=50
agg["bayes"]=(agg["count"]*agg["mean"]+mq*C)/(agg["count"]+mq)
print("C=",C)
print(agg.sort_values("bayes",ascending=False).head(10)[["Book-Title","Book-Author","count","mean","bayes"]].to_string())
print("raw mean top (count>=1):")
print(agg.sort_values(["mean","count"],ascending=False).head(5)[["Book-Title","count","mean"]].to_string())
# Harry Potter editions
hp=books[books["Book-Title"].str.contains("Harry Potter and the Sorcerer's Stone",regex=False)]
print("HP Sorcerer's stone ISBNs in books:",len(hp))
hpr=m[m.ISBN.isin(hp.ISBN)].groupby("ISBN").size().sort_values(ascending=False)
print(hpr.head(10).to_dict(), "total:",hpr.sum())
# item co-occurrence feasibility: books with >=10 users in matched data
bc=m.groupby("ISBN").size()
print("matched books with >=10 interactions:",(bc>=10).sum(), "they cover share of interactions:", bc[bc>=10].sum()/len(m))
print("matched books with >=50:",(bc>=50).sum())
