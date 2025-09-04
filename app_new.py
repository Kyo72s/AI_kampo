# app_new.py
# 修正点：
# - st.image を use_container_width=True に変更（赤い注意を解消）
# - 背景色を #D7FFB6 に統一（ページ全体に強制適用）
# - フォーム送信時の st.rerun() を削除して、送信後の二重リロードを防止

import os, re, unicodedata, datetime as dt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

APP_TITLE = "AI漢方選人"
TOP_N = 5
PCT_GAP_THRESHOLD = 0.30      # 1位との差が30%未満 → 追加質問対象
FOLLOWUP_PAGE_SIZE = 3        # 追加質問：各候補1ページあたり件数
W_MAIN = 2                    # 主症状重み
W_SUB  = 1                    # 他症状重み
PLANS  = ["Lite", "Standard", "Premium"]

# ============== 初期セットアップ ==============
load_dotenv()
st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")

CUSTOM_CSS = """
<style>
/* 背景を #D7FFB6 に統一（全レイヤーに強制） */
html, body, .stApp { background: #D7FFB6 !important; }
[data-testid="stAppViewContainer"] { background: #D7FFB6 !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* 幅広（1.5倍） */
.block-container { max-width: 1740px !important; }

:root { --card:#ffffff; --ink:#0f172a; --muted:#6b7280; --stroke:#e5e7eb; }
.small { color: var(--muted); font-size: 12px; }

section.card {
  background:var(--card); border:1px solid var(--stroke); border-radius:16px;
  padding:18px 20px; margin: 12px auto; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.kv { margin:10px 0 2px; font-weight:600; color:#111827; }
.kv + div { margin-bottom:10px; color:#111827; white-space:pre-wrap; }
hr { border:none; border-top:1px solid var(--stroke); margin:16px 0; }
.stButton > button[kind="primary"] { background:#ef4444; border-color:#ef4444; color:#fff; }
.stButton > button[kind="primary"]:hover { background:#dc2626; border-color:#dc2626; }
.cand > button { width:100%; text-align:left; border:1px solid var(--stroke)!important; border-radius:12px!important; padding:10px 14px!important; background:#fff!important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============== CSV 読み込みユーティリティ ==============
@st.cache_data
def load_any_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    for enc in (None, "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
        except Exception:
            pass
    return pd.read_csv(path, encoding="utf-8")

@st.cache_data
def load_data():
    sym  = load_any_csv("data/symptom_map.csv")
    main = load_any_csv("data/main_map.csv")
    km   = load_any_csv("data/kampo_master.csv")
    pm   = load_any_csv("data/product_master.csv")
    for df in (sym, main, km, pm):
        if df.empty: continue
        for c in df.columns:
            if df[c].dtype == object: df[c] = df[c].astype(str).str.strip()
    return sym, main, km, pm

symptom_map_raw, main_map_raw, kampo_master, product_master = load_data()

# ============== 正規化＆分割ユーティリティ ==============
def kana_to_hira(s: str) -> str:
    return "".join(chr(ord(ch)-0x60) if 0x30A1<=ord(ch)<=0x30F6 else ch for ch in s)

def normalize_ja(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s)
    s = kana_to_hira(s.lower()).replace("ー","").replace("・"," ")
    s = re.sub(r"\s+", "", s)
    return s

SYNONYMS = {
    "はきけ": {"吐き気","はきけ","嘔気","むかむか"},
    "おうと": {"嘔吐","吐く","吐いた"},
    "のどのかわき": {"口渇","のどが渇く","喉の渇き"},
    "ずつう": {"頭痛","頭が痛い"},
    "だるさ": {"だるい","倦怠感"},
}
def unify_synonym(token: str) -> str:
    n = normalize_ja(token)
    for canon, group in SYNONYMS.items():
        if n in {normalize_ja(x) for x in group}:
            return canon
    return n

def split_multi(text: str):
    """ 半角/全角スペース・改行・「、」「，」「,」「/」「／」を区切り """
    if not isinstance(text, str): return []
    parts = re.split(r"[,\s、，/／]+", text)
    return [p.strip() for p in parts if p.strip()]

def split_symptoms_cell(cell: str):
    if not isinstance(cell, str): return []
    t = cell.replace("、", ",").replace("，", ",").replace("・", ",").replace("／", ",").replace("/", ",")
    parts = [p.strip() for p in t.split(",") if p.strip()]
    if len(parts) == 1 and " " in parts[0]:
        parts = [p.strip() for p in parts[0].split() if p.strip()]
    return parts

def build_sets_both(df: pd.DataFrame, text_col_candidates=("症状","主症状","キーワード")):
    norm_dict, raw_dict = {}, {}
    if df.empty: return norm_dict, raw_dict
    name_col = "略称" if "略称" in df.columns else df.columns[0]
    text_col = None
    for c in text_col_candidates:
        if c in df.columns: text_col = c; break
    if text_col is None:
        text_col = df.columns[1] if len(df.columns)>1 else name_col
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        items_raw  = split_symptoms_cell(str(row[text_col]).strip())
        items_norm = {unify_synonym(x) for x in items_raw if x}
        norm_dict[name] = norm_dict.get(name,set()) | items_norm
        cur = raw_dict.get(name, [])
        seen = set(cur)
        for x in items_raw:
            if x not in seen:
                cur.append(x); seen.add(x)
        raw_dict[name] = cur
    return norm_dict, raw_dict

symptom_norm_sets, symptom_raw_lists = build_sets_both(symptom_map_raw, ("症状","キーワード","主症状"))
main_norm_sets, _                   = build_sets_both(main_map_raw, ("主症状","症状","キーワード"))

# ============== マッチング & 追加質問 ==============
def score_candidates(main_text: str, sub_text: str):
    toks_main = [unify_synonym(x) for x in split_multi(main_text)]
    toks_sub  = [unify_synonym(x) for x in split_multi(sub_text)]
    main_hits = set()
    if toks_main and main_norm_sets:
        for name, s in main_norm_sets.items():
            if any(t in s for t in toks_main): main_hits.add(name)
    pool_names = list(main_hits if main_hits else symptom_norm_sets.keys())
    rows=[]
    for name in pool_names:
        s_all_norm  = symptom_norm_sets.get(name, set())
        s_main_norm = main_norm_sets.get(name, set())
        base_hits   = sum(1 for kw in toks_sub  if kw in s_all_norm)
        main_hits2  = sum(1 for kw in toks_main if kw in s_main_norm)
        score       = W_SUB*base_hits + W_MAIN*main_hits2
        if score>0: rows.append({"略称":name,"score":score,"base":base_hits,"main":main_hits2})
    if not rows: return []
    df = pd.DataFrame(rows).sort_values(["score","main","base"], ascending=[False,False,False]).head(TOP_N)
    return df.to_dict("records")

def pct_gap_large_enough(cands, threshold=PCT_GAP_THRESHOLD):
    if len(cands)<2: return True
    s1,s2=cands[0]["score"], cands[1]["score"]
    if s1<=0: return True
    return (1.0 - s2/s1) >= threshold

def all_entered_tokens_norm() -> set[str]:
    toks = [*split_multi(st.session_state.get("main_text","")), *split_multi(st.session_state.get("sub_text",""))]
    return {unify_synonym(x) for x in toks if x}

def unique_per_candidate_within_group_raw(group_names):
    entered_norm = all_entered_tokens_norm()
    group_norm = {n: symptom_norm_sets.get(n,set()) for n in group_names}
    group_raw  = {n: symptom_raw_lists.get(n,[])  for n in group_names}
    result={}
    for n in group_names:
        others = set().union(*[group_norm[o] for o in group_names if o!=n])
        uniq_norm = group_norm[n] - others
        uniq_raw  = [raw for raw in group_raw[n]
                     if (unify_synonym(raw) in uniq_norm) and (unify_synonym(raw) not in entered_norm)]
        uniq_raw_sorted = sorted(uniq_raw, key=len)
        cleaned=[]
        for x in uniq_raw_sorted:
            if not any(x in y and x!=y for y in uniq_raw_sorted):
                cleaned.append(x)
        result[n]=cleaned
    return result

def target_group(cands):
    if not cands: return []
    s1 = cands[0]["score"]
    names=[cands[0]["略称"]]
    for c in cands[1:]:
        if s1>0 and (c["score"]/s1) > (1.0 - PCT_GAP_THRESHOLD):
            names.append(c["略称"])
    return names

def page_slice_dict(dict_lists, page, size):
    out={}; more=False
    for name, items in dict_lists.items():
        start=page*size; end=start+size
        out[name]=items[start:end]
        if len(items)>end: more=True
    return out, more

def pretty_text_common(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return "特になし"
    s=str(v)
    s=re.sub(r"(?:<br\s*/?>|\[\[BR\]\]|\\n|⏎|＜改行＞|<改行>)","\n",s,flags=re.IGNORECASE)
    s=re.sub(r"。[ \t\u3000]*","。\n",s)
    s=re.sub(r"[ \t\u3000]{2,}"," ",s)
    return s.strip() if s.strip() else "特になし"

def pretty_text_product(v, field_name: str):
    s=str(v)
    s=re.sub(r"(?:<br\s*/?>|\[\[BR\]\]|\\n|⏎|＜改行＞|<改行>)","\n",s,flags=re.IGNORECASE)
    if field_name == "組成":
        for kw in [r"日局", r"より製した", r"上記", r"以上の", r"本剤7\.5g中、\s*上記の"]:
            s = re.sub(rf"[ \t\u3000]*(?={kw})", "\n", s)
        s = re.sub(r"(?<=g)[ \t\u3000]*", "\n", s)
        s = re.sub(r"。\s*", "。\n", s)
    else:
        s = re.sub(r"。[ \t\u3000]*","。\n",s)
    s=re.sub(r"[ \t\u3000]{2,}"," ",s)
    return s.strip()

def render_kampo_detail(kampo_name: str):
    st.markdown(f"## {kampo_name}")
    km = kampo_master[kampo_master["略称"].astype(str)==kampo_name] if "略称" in kampo_master.columns else pd.DataFrame()
    with st.container():
        st.markdown("### 漢方解説")
        if km.empty:
            st.info("漢方薬マスタに該当データがありません。")
        else:
            preferred = ["ふりがな","出典","証","六病位","脈","舌","腹","漢方弁証","中医弁証"]
            hide_cols = {"略称","症状","漢方薬の事典の並び方"}
            cols = [c for c in preferred if c in km.columns] + [c for c in km.columns if c not in preferred and c not in hide_cols]
            row = km.iloc[0]
            for c in cols:
                st.markdown(f"<div class='kv'>{c}</div>", unsafe_allow_html=True)
                st.markdown(f"<div>{pretty_text_common(row[c])}</div>", unsafe_allow_html=True)

    if set(["略称","商品名"]).issubset(product_master.columns):
        pm = product_master[product_master["略称"].astype(str)==kampo_name]
        st.markdown("### 保険収載漢方エキス製剤一覧")
        st.markdown("<div class='small'>製剤名をクリックで添付文書情報を表示</div>", unsafe_allow_html=True)
        if pm.empty:
            st.info("該当製品は登録されていません。")
        else:
            for i, prod_name in enumerate(pm["商品名"].dropna().astype(str).unique().tolist(), start=1):
                if st.button(f"・{prod_name}", key=f"prod_btn_{kampo_name}_{i}", use_container_width=True):
                    st.session_state["selected_product"] = prod_name
                    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

    if st.session_state.get("selected_product"):
        render_product_detail(kampo_name, st.session_state["selected_product"])

def render_product_detail(kampo_name: str, product_name: str):
    pm = product_master
    pm = pm[(pm["略称"].astype(str)==kampo_name) & (pm["商品名"].astype(str)==product_name)]
    st.markdown(f"## {product_name}（製品詳細）")
    if pm.empty:
        st.info("該当製品が見つかりません。"); return
    row = pm.iloc[0]
    if "添付文書URL" in pm.columns and str(row.get("添付文書URL","")).startswith("http"):
        st.markdown(f"[添付文書を開く]({row['添付文書URL']})")
    display_map = {"商品番号": "一般的な製品番号"}
    for c in pm.columns:
        if c in ["略称","商品名","添付文書URL"]: continue
        label = display_map.get(c, c)
        st.markdown(f"<div class='kv'>{label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div>{pretty_text_product(row[c], c)}</div>", unsafe_allow_html=True)

# ============== 画面本体 ==============
left, center, right = st.columns([1,2,1])
with center:

    # ヘッダー画像（タイトル文字は表示しない）
    header_path = "AI_Kampo_sennin_title.png"  # プロジェクト直下に配置
    if os.path.exists(header_path):
        st.image(header_path, use_container_width=True)
    else:
        st.markdown("## " + APP_TITLE)  # フォールバック（画像がない場合のみ）

    # 右上：プラン + 無料トライアル（7日）
    col_title, col_plan = st.columns([1,1])
    with col_plan:
        st.selectbox("プラン", PLANS, key="plan")
        created = st.session_state.setdefault("created_at", dt.date.today())
        trial   = st.session_state.setdefault("trial_days", 7)  # 7日に固定
        remain_days = (dt.date.today() - created).days
        days_left   = max(0, trial - remain_days)
        st.caption(f"無料トライアル残り：{days_left}日")

    # 入力カード（form：1回送信でリロード1回のみに）
    st.markdown("<section class='card'>", unsafe_allow_html=True)
    with st.form(key="symptom_form", clear_on_submit=False):
        st.subheader("主症状")
        st.caption("最も気になる症状を1〜2語で入力してください（例：吐き気、頭痛）")
        main_input = st.text_input("主症状入力", key="form_main",
                                   value=st.session_state.get("main_text",""),
                                   placeholder="例：吐き気 頭痛", label_visibility="collapsed")

        st.subheader("他に気になる症状")
        st.caption("他に気になる症状・体質を自由に入力してください（例：めまい だるい 口渇 など）")
        sub_input  = st.text_area("他症状入力", key="form_sub",
                                  value=st.session_state.get("sub_text",""),
                                  height=90, placeholder="例：めまい だるい 口渇 など", label_visibility="collapsed")

        colS, colR = st.columns([1,1])
        submitted = colS.form_submit_button("送信", type="primary")
        if colR.form_submit_button("🔄 新しい漢方選びを始める"):
            st.session_state.update(main_text="", sub_text="", candidates=[],
                                   followup_page=0, selected_kampo=None, selected_product=None)
            # リセットは rerun して良い（操作の明確さ重視）
            st.rerun() if hasattr(st,"rerun") else st.experimental_rerun()
    st.markdown("</section>", unsafe_allow_html=True)

    # 送信処理：フォーム自体が1回リロードするので、ここでは rerun しない
    if submitted:
        st.session_state["main_text"]  = main_input
        st.session_state["sub_text"]   = sub_input
        st.session_state["candidates"] = score_candidates(main_input, sub_input)
        st.session_state["followup_page"] = 0
        st.session_state["selected_kampo"]  = None
        st.session_state["selected_product"]= None
        # ★ rerun しない：二重実行の原因になるため

    # 候補表示
    cands = st.session_state.get("candidates", [])
    if cands:
        with st.container():
            st.markdown("### AIによる処方提案（上位5件）")
            st.markdown("<div class='small'>漢方名をクリックで解説を表示</div>", unsafe_allow_html=True)
            top_score = max(c["score"] for c in cands) if cands else 1
            for i, c in enumerate(cands[:TOP_N], start=1):
                pct = int(round(95 * c["score"] / top_score)) if top_score>0 else 0
                pct = max(0, min(95, pct))
                if st.button(f"【{i}位】{c['略称']}（相性 {pct}%）", key=f"cand_{i}", use_container_width=True):
                    st.session_state["selected_kampo"]  = c["略称"]
                    st.session_state["selected_product"]= None
                    # ここは押した瞬間に詳細へ移るため rerun 維持
                    st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

        if not pct_gap_large_enough(cands, threshold=PCT_GAP_THRESHOLD):
            group = target_group(cands)
            uniq_dict_raw = unique_per_candidate_within_group_raw(group)
            page = st.session_state.get("followup_page", 0)
            sliced, more_exists = page_slice_dict(uniq_dict_raw, page, FOLLOWUP_PAGE_SIZE)

            st.markdown("**他にこんな症状や要素はありませんか？（該当があればその他の症状欄に追加入力してください）**")
            shown=False
            for name in group:
                items = sliced.get(name, [])
                if items:
                    shown=True
                    st.markdown(f"- **{name}** に特徴的： " + "、 ".join(items))
            if (not shown) and any(uniq_dict_raw.values()):
                more_exists = True
            if more_exists and st.button("さらに症状を提案する"):
                st.session_state["followup_page"] = page + 1
                st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

    # 漢方詳細（クリック後）
    if st.session_state.get("selected_kampo"):
        render_kampo_detail(st.session_state["selected_kampo"])
