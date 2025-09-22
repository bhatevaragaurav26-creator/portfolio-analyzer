from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import pandas as pd
from analytics import analyze

app = FastAPI(title="Portfolio Analyzer API")

class AnalyzeReq(BaseModel):
    tickers: List[str]
    weights: List[float]
    start: datetime
    end: datetime
    rf: float = 0.0
    var_cl: float = Field(0.95, ge=0.80, le=0.999)
    interval: str = "1d"

@app.post("/analyze")
def do_analyze(req: AnalyzeReq):
    prices, rets, port_ret, assets, summary = analyze(
        req.tickers, req.weights, req.start, req.end, rf=req.rf, cl=req.var_cl, interval=req.interval
    )
    return {
        "assets": assets,
        "summary": summary,
        "dates": [d.isoformat() for d in prices.index.to_pydatetime()],
        "prices": prices.fillna(None).to_dict(orient="list"),
        "port_ret": port_ret.fillna(0).tolist(),
    }
