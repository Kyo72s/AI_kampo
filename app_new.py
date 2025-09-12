# app_new.py
# 変更点：
# - 漢方解説の表示順を指定通りに変更（出典→証→六病位/虚実→脈→舌→腹→漢方弁証→中医弁証→一般的な製品番号）
# - 組成で必ず「日局」の直前で改行（(?<!\n)日局 → \n日局）を追加
# - 送信・リセット・候補/製剤クリックの挙動は前版のチューニングを維持（薄さ最小化）

import os, re, unicodedata, datetime as dt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

APP_TITLE = "AI漢方選人"
TOP_N = 5
PCT_GAP_THRESHOLD = 0.30
FOLLOWUP_PAGE_SIZE = 3
W_MAIN = 2
W_SUB  = 1
PLANS  = ["Lite", "Standard", "Premium"]

# ============== 初期セットアップ ==============
load_dotenv()
st.set_page_config(page_title=APP_TITLE, page_icon="💊", layout="wide")

CUSTOM_CSS = """
<style>
html, body, .stApp { background: #ecf7da !important; }
[data-testid="stAppViewContainer"] { background: #ecf7da !important; }
[data-testid="stHeader"] { background: transparent !important; }
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
    toks = [*split_multi(st.session_state.get("main_text","")),
            *split_multi(st.session_state.get("sub_text",""))]
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
    s = str(v)

    # 改行マーカーを統一（\n に）
    s = re.sub(r"(?:<br\s*/?>|\[\[BR\]\]|\\n|⏎|＜改行＞|<改行>)", "\n", s, flags=re.IGNORECASE)

    if field_name == "組成":
        text = s

        # 1) 「最初の生薬」の出現位置を探す（’日局’ または 和名＋数値＋g）
        ing_pat = re.compile(r"(日局|[一-龥ぁ-んァ-ンｦ-ﾟー]{1,12})\s*\d+(?:\.\d+)?g")
        m = ing_pat.search(text)
        if m:
            intro, rest = text[:m.start()], text[m.start():]
        else:
            intro, rest = text, ""

        # 2) 冒頭説明：句点のみで改行。その他の改行は除去して1行に整える
        intro = intro.replace("\n", " ")
        intro = re.sub(r"。\s*", "。<br/>", intro)

        # 3) 成分ブロック
        rest = rest.replace("\n", " ")

        # 3-1) 全ての「日局」の直前に改行
        rest = rest.replace("日局", "<br/>日局")

        # 3-2) 「日局なし生薬」でも改行（エキス/粉末は除外）
        #     直前数文字に「エキス」「粉末」がない場合のみ改行を入れる
        #     例:  … シンキク2.0g → <br/>シンキク2.0g
        rest = re.sub(
            r"(?<!エキス)(?<!粉末)\s([一-龥ぁ-んァ-ンｦ-ﾟー]{1,12})\s*([0-9]+(?:\.[0-9]+)?g)",
            r"<br/>\1\2",
            rest
        )

        # 3-3) 連続 <br/> を 1 つに
        rest = re.sub(r"(?:<br/>\s*){2,}", "<br/>", rest).lstrip("<br/>")

        # 4) 結合
        out = (intro + rest).strip()

        # 5) 余分な空白整理
        out = re.sub(r"[ \t\u3000]{2,}", " ", out)

        return out

    else:
        # 組成以外は従来通り：句点で改行 → <br/> へ
        s = re.sub(r"。[ \t\u3000]*", "。\n", s)
        s = s.replace("\n", "<br/>")
        s = re.sub(r"[ \t\u3000]{2,}", " ", s)
        return s.strip()



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

def render_kampo_detail(kampo_name: str):
    """CSV(kampo_master.csv)の列名を尊重した順で漢方解説を表示する"""
    st.markdown(f"## {kampo_name}")

    km = kampo_master[kampo_master["略称"].astype(str) == kampo_name] if "略称" in kampo_master.columns else pd.DataFrame()
    with st.container():
        st.markdown("### 漢方解説")
        if km.empty:
            st.info("漢方薬マスタに該当データがありません。")
            return

        row_dict = km.iloc[0].to_dict()

        # 列名の空白ゆらぎを吸収（全角/半角スペースをすべて除去して比較）
        def norm_key(s: str) -> str:
            return re.sub(r"\s+", "", str(s))

        # 実列名を安全に見つける
        def get_val_by_label(label: str) -> str:
            target = norm_key(label)
            for k, v in row_dict.items():
                if norm_key(k) == target:
                    return "" if v is None else str(v).strip()
            return ""

        # 値だけ表示（ラベル無し）
        def show_value_only(label: str):
            v = get_val_by_label(label)
            if v:
                st.markdown(f"{pretty_text_common(v)}")

        # ラベル＋値
        def show_labeled(label: str):
            v = get_val_by_label(label)
            if v:
                st.markdown(f"<div class='kv'>{label}</div>", unsafe_allow_html=True)
                st.markdown(f"<div>{pretty_text_common(v)}</div>", unsafe_allow_html=True)

        # 1) 略称・ふりがな はラベル無しで上に表示
        show_value_only("略称")
        show_value_only("ふりがな")

        # 2) 以降はCSVの列名どおり、指定順に“そのまま”表示
        ordered_labels = [
            "出典",
            "証（表裏・寒熱・虚実）",
            "六病位 ／  虚実",   # 空白の有無は norm_key で無視して突き合わせます
            "脈",
            "舌",
            "腹",
            "漢方弁証",
            "中医弁証",
        ]
        for lab in ordered_labels:
            show_labeled(lab)

    # === 製剤一覧（従来どおり） ===
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

    if st.session_state.get("selected_product"):
        render_product_detail(kampo_name, st.session_state["selected_product"])


# ============== 画面本体（薄さ最小化のロジックは前版のまま） ==============
left, center, right = st.columns([1,2,1])
with center:
    # 先頭でリセット処理（Widget生成前に）
    if st.session_state.get("_do_reset"):
        for k in ["form_main","form_sub","main_text","sub_text","candidates",
                  "followup_page","selected_kampo","selected_product"]:
            st.session_state.pop(k, None)
        st.session_state["_do_reset"] = False

    header_path = "AI_Kampo_sennin_title.png"
    if os.path.exists(header_path):
        st.image(header_path, use_container_width=True)
    else:
        st.markdown("## " + APP_TITLE)

    if st.button("🔄 新しい漢方選びを始める", help="入力欄・候補をクリアします"):
        st.session_state["_do_reset"] = True
        st.rerun() if hasattr(st,"rerun") else st.experimental_rerun()

    col_title, col_plan = st.columns([1,1])
    with col_plan:
        st.selectbox("プラン", PLANS, key="plan")
        created = st.session_state.setdefault("created_at", dt.date.today())
        trial   = st.session_state.setdefault("trial_days", 7)
        remain_days = (dt.date.today() - created).days
        days_left   = max(0, trial - remain_days)
        st.caption(f"無料トライアル残り：{days_left}日")

    if "form_main" not in st.session_state:
        st.session_state["form_main"] = st.session_state.get("main_text","")
    if "form_sub" not in st.session_state:
        st.session_state["form_sub"]  = st.session_state.get("sub_text","")

    st.markdown("<section class='card'>", unsafe_allow_html=True)
    with st.form("symptom_form", clear_on_submit=False):
        st.subheader("主症状")
        st.caption("最も気になる症状を1〜2語で入力してください（例：吐き気、頭痛）")
        st.text_input("主症状入力", key="form_main",
                      placeholder="例：吐き気 頭痛", label_visibility="collapsed")

        st.subheader("他に気になる症状")
        st.caption("他に気になる症状・体質を自由に入力してください（例：めまい だるい 口渇 など）")
        st.text_area("他症状入力", key="form_sub", height=90,
                     placeholder="例：めまい だるい 口渇 など", label_visibility="collapsed")

        submitted = st.form_submit_button("送信", type="primary")
    st.markdown("</section>", unsafe_allow_html=True)

    new_cands = None
    if submitted:
        main_input = st.session_state.get("form_main","")
        sub_input  = st.session_state.get("form_sub","")
        st.session_state["main_text"]  = main_input
        st.session_state["sub_text"]   = sub_input
        new_cands = score_candidates(main_input, sub_input)
        st.session_state["candidates"] = new_cands
        st.session_state["followup_page"] = 0
        st.session_state["selected_kampo"]  = None
        st.session_state["selected_product"]= None

    cands = new_cands if new_cands is not None else st.session_state.get("candidates", [])
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

    if st.session_state.get("selected_kampo"):
        render_kampo_detail(st.session_state["selected_kampo"])









