# AI馬券師推奨機能 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AIが全券種（単勝〜三連単）を評価し、1枚100円固定で最適な馬券を推奨するUI・ロジックを追加する。

**Architecture:** `value_ai.py` に `recommend_betmaster_plans()` / `select_primary_betmaster()` を追加し、`keiba_app.py` の推奨セクションを新UIに差し替える。bankroll 入力は 100円固定にコメントアウト。

**Tech Stack:** Python 3.14, Streamlit, LightGBM（既存）, itertools（標準ライブラリ）

---

## ファイルマップ

| ファイル | 変更内容 |
|---------|---------|
| `value_ai.py` | 定数追加 + `recommend_betmaster_plans()` + `select_primary_betmaster()` を末尾に追加 |
| `keiba_app.py` | import追加、bankroll固定化、推奨UIセクション差し替え（行1610〜1740） |
| `tests/test_betmaster.py` | 新規作成（ユニットテスト） |

---

## Task 1: value_ai.py に定数と補助関数を追加

**Files:**
- Modify: `value_ai.py`（末尾 `value_ai.py:2500` 付近に追記）
- Test: `tests/test_betmaster.py`（新規作成）

- [ ] **Step 1: テストファイルを新規作成し、失敗することを確認**

```bash
mkdir -p /Users/ryokarahashi/keiba_ai/tests
```

`tests/test_betmaster.py` を以下の内容で作成：

```python
"""tests/test_betmaster.py — AI馬券師推奨機能のユニットテスト"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from value_ai import recommend_betmaster_plans, select_primary_betmaster

# テスト用フィクスチャ: 18頭立て想定
FEATURES_18 = [
    {"horse_name": f"馬{i}", "win_prob": max(0.01, 0.25 - i * 0.013),
     "consistency_index": 0.7 - i * 0.03, "trend_index": 0.65 - i * 0.02,
     "popularity": i + 1}
    for i in range(18)
]

HORSE_ROLES_18 = [
    {"horse_name": "馬0", "role": "head"},
    {"horse_name": "馬1", "role": "axis"},
    {"horse_name": "馬2", "role": "axis"},
    {"horse_name": "馬3", "role": "himo"},
    {"horse_name": "馬4", "role": "himo"},
    {"horse_name": "馬5", "role": "himo"},
    {"horse_name": "馬6", "role": "fade"},
] + [{"horse_name": f"馬{i}", "role": "himo"} for i in range(7, 18)]

RACE_STRUCTURE = {"structure_type": "標準型", "favorable_style": "front"}


def test_recommend_betmaster_plans_returns_nine_types():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    bet_types = [p["bet_type"] for p in plans]
    assert "単勝" in bet_types
    assert "複勝" in bet_types
    assert "ワイド" in bet_types
    assert "馬連（流し）" in bet_types
    assert "馬単フォーメーション" in bet_types
    assert "三連複フォーメーション（AI絞り）" in bet_types
    assert "三連複フォーメーション（全頭）" in bet_types
    assert "三連単フォーメーション（AI絞り）" in bet_types
    assert "三連単フォーメーション（全頭）" in bet_types


def test_all_tickets_are_100_yen():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    for plan in plans:
        if plan["confidence_ok"]:
            for t in plan["tickets"]:
                assert t["stake"] == 100, f"{plan['bet_type']} has stake {t['stake']}"


def test_budget_equals_ticket_count_times_100():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    for plan in plans:
        if plan["confidence_ok"]:
            assert plan["budget"] == plan["ticket_count"] * 100


def test_no_fade_horse_in_tickets():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    for plan in plans:
        if plan["confidence_ok"]:
            for t in plan["tickets"]:
                assert "馬6" not in t["combination"], \
                    f"fade馬が {plan['bet_type']} に含まれている"


def test_formation_legs_present_for_multi_leg_bets():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    for plan in plans:
        if "フォーメーション" in plan["bet_type"] or "流し" in plan["bet_type"]:
            assert plan["formation_legs"] is not None, \
                f"{plan['bet_type']} に formation_legs がない"


def test_select_primary_betmaster_returns_one():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    primary = select_primary_betmaster(plans, RACE_STRUCTURE)
    assert primary is not None
    assert primary["confidence_ok"] is True


def test_fallback_no_roles():
    """horse_roles なしでも動作する"""
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, horse_roles=None)
    assert len(plans) == 9
    for plan in plans:
        assert "bet_type" in plan
        assert "confidence_ok" in plan
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python -m pytest tests/test_betmaster.py -v 2>&1 | head -30
```

期待: `ImportError: cannot import name 'recommend_betmaster_plans'`

---

## Task 2: value_ai.py に `recommend_betmaster_plans()` を実装

**Files:**
- Modify: `value_ai.py`（末尾に追記）

- [ ] **Step 1: 定数と補助関数を value_ai.py 末尾に追記**

`value_ai.py` の末尾（2500行目付近）に以下を追記：

```python
# =========================================================
# AI馬券師推奨機能 — 全券種・1枚100円固定
# =========================================================

BETMASTER_TICKET_UNIT: int = 100  # 1枚100円固定

# 自信度閾値
CONFIDENCE_TANSHO: float = 0.60    # 単勝: head の stable_score
CONFIDENCE_FUKUSHO: float = 0.50   # 複勝: head の top3_prob（estimate_placement_probs から）
CONFIDENCE_BATAN: float = 0.65     # 馬単・三連単: head の stable_score（厳しめ）


def _bm_stable(f: Dict[str, Any]) -> float:
    """stable_score = (consistency_index + trend_index) / 2"""
    return (float(f.get("consistency_index") or 0.5) + float(f.get("trend_index") or 0.5)) / 2.0


def _bm_sorted_candidates(
    features: List[Dict[str, Any]],
    horse_roles: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    horse_roles に従って馬をソートし、role_map も返す。
    horse_roles が None の場合は win_prob 順で自動割当。
    fade 馬は non_fade リストから除外する。
    """
    _ROLE_ORDER = {"head": 0, "axis": 1, "himo": 2, "fade": 3}

    if horse_roles:
        role_map: Dict[str, str] = {r["horse_name"]: r.get("role", "himo") for r in horse_roles}
        sorted_f = sorted(
            features,
            key=lambda x: (
                _ROLE_ORDER.get(role_map.get(str(x.get("horse_name") or ""), "himo"), 2),
                -float(x.get("win_prob") or 0.0),
            ),
        )
        non_fade = [f for f in sorted_f
                    if role_map.get(str(f.get("horse_name") or ""), "himo") != "fade"]
    else:
        sorted_f = sorted(features, key=lambda x: -float(x.get("win_prob") or 0.0))
        non_fade = sorted_f
        role_map = {}
        for i, f in enumerate(non_fade):
            name = str(f.get("horse_name") or "")
            if i == 0:
                role_map[name] = "head"
            elif i <= 2:
                role_map[name] = "axis"
            else:
                role_map[name] = "himo"

    # non_fade が2頭未満の場合は fade から補充
    if len(non_fade) < 2:
        fade_horses = [f for f in sorted_f if f not in non_fade]
        non_fade = non_fade + fade_horses[:2 - len(non_fade)]

    return non_fade, role_map


def _bm_formation_trio_tickets(
    leg1: List[str], leg2: List[str], leg3: List[str]
) -> List[Dict[str, Any]]:
    """
    三連複フォーメーション: leg1/leg2/leg3 から3頭の組み合わせを生成。
    各レグから少なくとも1頭を含み、重複しないユニークな3頭組み合わせを返す。
    """
    seen: Set[frozenset] = set()
    tickets: List[Dict[str, Any]] = []
    for a in leg1:
        for b in leg2:
            for c in leg3:
                combo = frozenset([a, b, c])
                if len(combo) == 3 and combo not in seen:
                    seen.add(combo)
                    tickets.append({"combination": sorted([a, b, c]), "stake": BETMASTER_TICKET_UNIT})
    return tickets


def _bm_formation_trifecta_tickets(
    leg1: List[str], leg2: List[str], leg3: List[str]
) -> List[Dict[str, Any]]:
    """
    三連単フォーメーション: (1着, 2着, 3着) の順列を生成。
    各馬は1回のみ使用。
    """
    seen: Set[Tuple[str, str, str]] = set()
    tickets: List[Dict[str, Any]] = []
    for a in leg1:
        for b in leg2:
            for c in leg3:
                if len({a, b, c}) == 3 and (a, b, c) not in seen:
                    seen.add((a, b, c))
                    tickets.append({"combination": [a, b, c], "stake": BETMASTER_TICKET_UNIT})
    return tickets


def _bm_plan(
    bet_type: str,
    formation_legs: Optional[Dict[str, List[str]]],
    tickets: List[Dict[str, Any]],
    risk_level: str,
    reason: str,
    confidence_ok: bool,
    no_pick_reason: str,
    confidence_score: float,
) -> Dict[str, Any]:
    """AI馬券師プランの標準辞書を返す。"""
    count = len(tickets)
    return {
        "bet_type":        bet_type,
        "formation_legs":  formation_legs,
        "tickets":         tickets,
        "ticket_count":    count,
        "budget":          count * BETMASTER_TICKET_UNIT,
        "risk_level":      risk_level,
        "reason":          reason,
        "confidence_ok":   confidence_ok,
        "no_pick_reason":  no_pick_reason,
        "confidence_score": confidence_score,
    }
```

- [ ] **Step 2: `recommend_betmaster_plans()` 本体を追記**

同じく `value_ai.py` 末尾に続けて追記：

```python
def recommend_betmaster_plans(
    features: List[Dict[str, Any]],
    race_structure: Dict[str, Any],
    horse_roles: Optional[List[Dict[str, Any]]] = None,
    race_pace: str = "medium",
) -> List[Dict[str, Any]]:
    """
    全券種（単勝・複勝・ワイド・馬連・馬単・三連複×2・三連単×2）を評価し、
    AI馬券師として1枚100円固定で推奨する買い目リストを返す。

    各プランに confidence_ok フラグを付与し、
    自信がない券種は tickets=[] / confidence_ok=False / no_pick_reason=理由 を返す。
    """
    from value_ai import estimate_placement_probs  # 同一モジュール内参照

    non_fade, role_map = _bm_sorted_candidates(features, horse_roles)
    if not non_fade:
        return []

    # 役割別に馬を分類
    head_horses  = [f for f in non_fade if role_map.get(str(f.get("horse_name") or ""), "himo") == "head"]
    axis_horses  = [f for f in non_fade if role_map.get(str(f.get("horse_name") or ""), "himo") == "axis"]
    himo_horses  = [f for f in non_fade if role_map.get(str(f.get("horse_name") or ""), "himo") == "himo"]

    # head がいない場合は win_prob 最上位を head として使う
    if not head_horses:
        head_horses = non_fade[:1]
    if not axis_horses:
        axis_horses = non_fade[1:3]
    if not himo_horses:
        himo_horses = non_fade[3:]

    head = head_horses[0]
    head_name = str(head.get("horse_name") or "")
    head_stable = _bm_stable(head)

    # head の top3_prob（estimate_placement_probs で計算）
    pp = estimate_placement_probs(head, race_structure)
    head_top3_prob = pp["p_top3"]
    head_win_prob  = float(head.get("win_prob") or 0.0)

    axis_names  = [str(f.get("horse_name") or "") for f in axis_horses[:3]]
    himo_names  = [str(f.get("horse_name") or "") for f in himo_horses]
    all_names   = [str(f.get("horse_name") or "") for f in non_fade]

    plans: List[Dict[str, Any]] = []

    # ── 1. 単勝 ───────────────────────────────────────────────────────────
    ok_tansho = head_stable >= CONFIDENCE_TANSHO
    plans.append(_bm_plan(
        bet_type="単勝",
        formation_legs=None,
        tickets=[{"combination": [head_name], "stake": BETMASTER_TICKET_UNIT}] if ok_tansho else [],
        risk_level="低",
        reason=f"{head_name} の安定指数 {head_stable:.2f}（閾値{CONFIDENCE_TANSHO}）",
        confidence_ok=ok_tansho,
        no_pick_reason="" if ok_tansho else f"軸馬の安定指数不足（{head_stable:.2f} < {CONFIDENCE_TANSHO}）",
        confidence_score=head_stable,
    ))

    # ── 2. 複勝 ───────────────────────────────────────────────────────────
    ok_fukusho = head_top3_prob >= CONFIDENCE_FUKUSHO
    plans.append(_bm_plan(
        bet_type="複勝",
        formation_legs=None,
        tickets=[{"combination": [head_name], "stake": BETMASTER_TICKET_UNIT}] if ok_fukusho else [],
        risk_level="最低",
        reason=f"{head_name} の3着圏内確率 {head_top3_prob:.2f}（閾値{CONFIDENCE_FUKUSHO}）",
        confidence_ok=ok_fukusho,
        no_pick_reason="" if ok_fukusho else f"3着圏内確率不足（{head_top3_prob:.2f} < {CONFIDENCE_FUKUSHO}）",
        confidence_score=head_top3_prob,
    ))

    # ── 3. ワイド ────────────────────────────────────────────────────────
    wide_legs = [head_name] + axis_names[:2]
    wide_combos = list(combinations(wide_legs, 2))
    wide_tickets = [{"combination": list(c), "stake": BETMASTER_TICKET_UNIT} for c in wide_combos]
    plans.append(_bm_plan(
        bet_type="ワイド",
        formation_legs={"組み合わせ": wide_legs},
        tickets=wide_tickets,
        risk_level="低",
        reason=f"能力上位{len(wide_legs)}頭のワイドBOX",
        confidence_ok=True,
        no_pick_reason="",
        confidence_score=head_stable,
    ))

    # ── 4. 馬連（流し）────────────────────────────────────────────────────
    umaren_partners = axis_names + [str(f.get("horse_name") or "") for f in himo_horses[:3]]
    umaren_partners = [n for n in umaren_partners if n != head_name][:5]
    umaren_tickets = [{"combination": sorted([head_name, p]), "stake": BETMASTER_TICKET_UNIT}
                      for p in umaren_partners]
    plans.append(_bm_plan(
        bet_type="馬連（流し）",
        formation_legs={"軸": [head_name], "相手": umaren_partners},
        tickets=umaren_tickets,
        risk_level="低",
        reason=f"{head_name} 軸・馬連流し {len(umaren_tickets)}点",
        confidence_ok=True,
        no_pick_reason="",
        confidence_score=head_stable,
    ))

    # ── 5. 馬単フォーメーション ──────────────────────────────────────────
    ok_batan = head_stable >= CONFIDENCE_BATAN
    batan_partners = axis_names[:3]
    batan_tickets = [{"combination": [head_name, p], "stake": BETMASTER_TICKET_UNIT}
                     for p in batan_partners] if ok_batan else []
    plans.append(_bm_plan(
        bet_type="馬単フォーメーション",
        formation_legs={"1着": [head_name], "2着": batan_partners} if batan_partners else None,
        tickets=batan_tickets,
        risk_level="中",
        reason=f"{head_name} 1着固定・馬単 {len(batan_tickets)}点",
        confidence_ok=ok_batan,
        no_pick_reason="" if ok_batan else f"1着固定の確度不足（安定指数 {head_stable:.2f} < {CONFIDENCE_BATAN}）",
        confidence_score=head_stable,
    ))

    # ── 6. 三連複フォーメーション（AI絞り）────────────────────────────────
    trio_leg1 = [head_name]
    trio_leg2 = axis_names[:2]
    trio_leg3_ai = [str(f.get("horse_name") or "") for f in
                    sorted(himo_horses, key=lambda x: -float(x.get("win_prob") or 0.0))[:6]]
    # leg3 に leg2 の馬が被ることを許容（フォーメーション上正常）
    trio_ai_tickets = _bm_formation_trio_tickets(trio_leg1, trio_leg2, trio_leg3_ai)
    plans.append(_bm_plan(
        bet_type="三連複フォーメーション（AI絞り）",
        formation_legs={"馬1": trio_leg1, "馬2": trio_leg2, "馬3": trio_leg3_ai},
        tickets=trio_ai_tickets,
        risk_level="中",
        reason=f"馬1:{head_name} / 馬2:{len(trio_leg2)}頭 / 馬3:AI絞り{len(trio_leg3_ai)}頭",
        confidence_ok=True,
        no_pick_reason="",
        confidence_score=head_stable,
    ))

    # ── 7. 三連複フォーメーション（全頭）─────────────────────────────────
    trio_leg3_all = [n for n in all_names if n != head_name and n not in trio_leg2]
    trio_all_tickets = _bm_formation_trio_tickets(trio_leg1, trio_leg2, trio_leg3_all)
    plans.append(_bm_plan(
        bet_type="三連複フォーメーション（全頭）",
        formation_legs={"馬1": trio_leg1, "馬2": trio_leg2, "馬3": trio_leg3_all},
        tickets=trio_all_tickets,
        risk_level="中",
        reason=f"馬1:{head_name} / 馬2:{len(trio_leg2)}頭 / 馬3:全{len(trio_leg3_all)}頭",
        confidence_ok=True,
        no_pick_reason="",
        confidence_score=head_stable,
    ))

    # ── 8. 三連単フォーメーション（AI絞り）────────────────────────────────
    trifecta_leg1 = [head_name]
    trifecta_leg2 = axis_names[:3]
    trifecta_leg3_ai = [str(f.get("horse_name") or "") for f in
                        sorted(himo_horses, key=lambda x: -float(x.get("win_prob") or 0.0))[:6]]
    trifecta_ai_tickets = _bm_formation_trifecta_tickets(
        trifecta_leg1, trifecta_leg2, trifecta_leg3_ai
    ) if ok_batan else []
    plans.append(_bm_plan(
        bet_type="三連単フォーメーション（AI絞り）",
        formation_legs={"1着": trifecta_leg1, "2着": trifecta_leg2, "3着": trifecta_leg3_ai},
        tickets=trifecta_ai_tickets,
        risk_level="高",
        reason=f"1着:{head_name} / 2着:{len(trifecta_leg2)}頭 / 3着:AI絞り{len(trifecta_leg3_ai)}頭",
        confidence_ok=ok_batan,
        no_pick_reason="" if ok_batan else f"1着固定の確度不足（安定指数 {head_stable:.2f} < {CONFIDENCE_BATAN}）",
        confidence_score=head_stable,
    ))

    # ── 9. 三連単フォーメーション（全頭）─────────────────────────────────
    trifecta_leg3_all = [n for n in all_names if n != head_name and n not in trifecta_leg2]
    trifecta_all_tickets = _bm_formation_trifecta_tickets(
        trifecta_leg1, trifecta_leg2, trifecta_leg3_all
    ) if ok_batan else []
    plans.append(_bm_plan(
        bet_type="三連単フォーメーション（全頭）",
        formation_legs={"1着": trifecta_leg1, "2着": trifecta_leg2, "3着": trifecta_leg3_all},
        tickets=trifecta_all_tickets,
        risk_level="高",
        reason=f"1着:{head_name} / 2着:{len(trifecta_leg2)}頭 / 3着:全{len(trifecta_leg3_all)}頭",
        confidence_ok=ok_batan,
        no_pick_reason="" if ok_batan else f"1着固定の確度不足（安定指数 {head_stable:.2f} < {CONFIDENCE_BATAN}）",
        confidence_score=head_stable,
    ))

    return plans
```

- [ ] **Step 3: テストを実行して確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python -m pytest tests/test_betmaster.py::test_recommend_betmaster_plans_returns_nine_types tests/test_betmaster.py::test_all_tickets_are_100_yen tests/test_betmaster.py::test_budget_equals_ticket_count_times_100 -v
```

期待: 3テスト PASS

---

## Task 3: value_ai.py に `select_primary_betmaster()` を実装

**Files:**
- Modify: `value_ai.py`（Task 2 の末尾に続けて追記）

- [ ] **Step 1: `select_primary_betmaster()` を追記**

```python
def select_primary_betmaster(
    plans: List[Dict[str, Any]],
    race_structure: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    recommend_betmaster_plans() の結果から主推奨を1つ選定して返す。
    confidence_ok=True の中からレース構造に基づいてスコアリングし最高値を返す。
    全て confidence_ok=False の場合は None を返す。
    """
    structure_type = (race_structure or {}).get("structure_type", "標準型")

    # 券種別の優先度スコア（レース構造に応じて変動）
    _BASE_PRIORITY: Dict[str, float] = {
        "単勝":                        0.50,
        "複勝":                        0.20,
        "ワイド":                       0.40,
        "馬連（流し）":                  0.55,
        "馬単フォーメーション":           0.60,
        "三連複フォーメーション（AI絞り）": 0.80,
        "三連複フォーメーション（全頭）":   0.65,
        "三連単フォーメーション（AI絞り）": 0.70,
        "三連単フォーメーション（全頭）":   0.55,
    }

    _STRUCTURE_BONUS: Dict[str, Dict[str, float]] = {
        "本命信頼型": {
            "馬単フォーメーション":           0.20,
            "三連単フォーメーション（AI絞り）": 0.15,
            "馬連（流し）":                  0.10,
        },
        "標準型": {
            "三連複フォーメーション（AI絞り）": 0.15,
            "馬連（流し）":                  0.05,
        },
        "1強相手混戦型": {
            "三連複フォーメーション（AI絞り）": 0.10,
            "三連複フォーメーション（全頭）":   0.15,
        },
        "混戦型": {
            "三連複フォーメーション（全頭）":   0.20,
            "ワイド":                       0.10,
        },
        "波乱型": {
            "ワイド":                       0.25,
            "複勝":                        0.15,
            "三連複フォーメーション（全頭）":   0.10,
        },
        "差し届く型": {
            "三連複フォーメーション（AI絞り）": 0.10,
            "三連単フォーメーション（AI絞り）": 0.10,
        },
    }

    bonus_map = _STRUCTURE_BONUS.get(structure_type, {})

    best_plan: Optional[Dict[str, Any]] = None
    best_score: float = -1.0

    for plan in plans:
        if not plan.get("confidence_ok"):
            continue
        if not plan.get("tickets"):
            continue

        base = _BASE_PRIORITY.get(plan["bet_type"], 0.4)
        bonus = bonus_map.get(plan["bet_type"], 0.0)
        # 点数が多すぎるとペナルティ（100点超は大幅減点）
        ticket_penalty = min(0.20, plan["ticket_count"] / 500.0)
        score = base + bonus - ticket_penalty

        if score > best_score:
            best_score = score
            best_plan = plan

    return best_plan
```

- [ ] **Step 2: テストを実行して確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python -m pytest tests/test_betmaster.py -v
```

期待: 全テスト PASS（7テスト）

- [ ] **Step 3: コミット**

```bash
cd /Users/ryokarahashi/keiba_ai && git add value_ai.py tests/test_betmaster.py && git commit -m "feat: add recommend_betmaster_plans and select_primary_betmaster to value_ai"
```

---

## Task 4: keiba_app.py のインポートと bankroll を更新

**Files:**
- Modify: `keiba_app.py`

- [ ] **Step 1: import に新関数を追加**

`keiba_app.py` の22〜35行目の import ブロックを以下に変更：

```python
from value_ai import (
    build_ev_table,
    build_ticket_ev_table,
    detect_value_horses,
    detect_danger_favorites_v3,
    classify_race_structure,
    recommend_bet_plan,
    recommend_all_bet_types,
    recommend_betmaster_plans,
    select_primary_betmaster,
    assign_marks,
    assign_roles,
    detect_rescue_candidates,
    VALUE_GAP_MIN,
    DANGER_GAP_MIN,
)
```

- [ ] **Step 2: bankroll 入力欄をコメントアウトし固定値に変更**

`keiba_app.py` の736行目を以下に変更：

```python
# bankroll = st.number_input("軍資金", min_value=100, value=1000, step=100)  # 将来復活予定
bankroll = 100  # 1枚100円固定
```

- [ ] **Step 3: アプリが起動することを確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python -c "import keiba_app" 2>&1 | head -20
```

期待: エラーなし（空出力 or StreamlitのDryRunWarning のみ）

- [ ] **Step 4: コミット**

```bash
cd /Users/ryokarahashi/keiba_ai && git add keiba_app.py && git commit -m "feat: fix bankroll to 100 yen, add betmaster imports"
```

---

## Task 5: keiba_app.py の推奨UIセクションを差し替え

**Files:**
- Modify: `keiba_app.py`（行1610〜1740 の「① 推奨買い目」セクションを丸ごと置換）

- [ ] **Step 1: 既存セクション（行1610〜1740）を新UIコードに置換**

置換対象（`keiba_app.py` の現在の内容）：
```python
        # ── ① 推奨買い目（最重要 — 最上部に表示） ───────────────────────
        st.subheader("推奨買い目")

        # 軸馬手動指定
        _all_horse_names = [f.get("horse_name", "") for f in result.get("features", []) if f.get("horse_name")]
        _pop_sorted_names = [
            f.get("horse_name", "") for f in sorted(
                result.get("features", []),
                key=lambda x: float(x.get("popularity") or 99),
            ) if f.get("horse_name")
        ]
        _forced_axis = st.selectbox(
            "軸馬を固定（任意）",
            options=["AIに任せる"] + _pop_sorted_names,
            index=0,
            key="forced_axis_select",
            help="選択した馬が必ず含まれる組み合わせのみ推奨します。上位人気を軸にしたい場合に使用。",
        )
        _forced_axis_name = None if _forced_axis == "AIに任せる" else _forced_axis

        ticket_evs_summary: list = []
        if not ev_table_ui:
            st.info("オッズを入力後に推奨買い目が表示されます。")
        else:
            ticket_evs_summary = build_ticket_ev_table(
                result.get("features", []),
                race_structure=race_structure,
                ev_table=ev_table_ui,
                forced_axis=_forced_axis_name,
                horse_roles=horse_roles,
            )
            best_by_type: dict = {}
            for r in ticket_evs_summary:
                bt = r["bet_type"]
                if bt not in best_by_type or r["ev"] > best_by_type[bt]["ev"]:
                    best_by_type[bt] = r

            bet_plan = recommend_bet_plan(
                result.get("features", []),
                ev_table_ui,
                race_structure,
                bankroll,
                race_pace_ev,
                forced_axis=_forced_axis_name,
                horse_roles=horse_roles,
            )

            if bet_plan.get("skip"):
                st.warning(f"🚫 見送り推奨")
                skip_reason = bet_plan.get("skip_reason", "")
                if skip_reason:
                    st.caption(skip_reason)
                if ticket_evs_summary:
                    with st.expander("参考: 全券種EV（閾値未達）", expanded=False):
                        st.caption("※ いずれも推奨閾値（単勝/複勝≥1.02、複合≥0.95）に届きませんでした。")
                        skip_rows = [{
                            "券種": r["bet_type"],
                            "馬": " / ".join(r["horses"]),
                            "EV(補正後)": f"{r['ev']:.3f}",
                        } for r in ticket_evs_summary[:8]]
                        st.dataframe(pd.DataFrame(skip_rows), use_container_width=True)
            else:
                bp_cols = st.columns(3)
                bp_cols[0].metric("券種", bet_plan.get("bet_type", "-"))
                bp_cols[1].metric("点数", f"{bet_plan.get('ticket_count', 0)}点")
                bp_cols[2].metric("合計金額", f"¥{bet_plan.get('total_stake', 0):,}")

                risk_col, ev_col = st.columns(2)
                risk_col.metric("リスクレベル", bet_plan.get("risk_level", "-"))
                ev_type = bet_plan.get("ev_type", "-")
                ev_col.metric("選定方式",
                    "📊 EV比較型" if ev_type == "EV比較型" else "🗺️ 構造型" if ev_type == "構造型" else ev_type)

                st.caption(f"根拠: {bet_plan.get('reason', '-')}")
                if ev_type == "EV比較型" and danger_v2:
                    excluded = [d["horse_name"] for d in danger_v2]
                    st.caption(f"⚠️ 危険人気馬（{' / '.join(excluded)}）を含む組み合わせは減点されています。")

                tickets = bet_plan.get("tickets", [])
                if tickets:
                    st.markdown("**買い目詳細**")
                    st.dataframe(pd.DataFrame([{
                        "組み合わせ": "・".join(t.get("combination", [])),
                        "金額": f"¥{t.get('stake', 0):,}",
                    } for t in tickets]), use_container_width=True, hide_index=True)

                sel_detail = bet_plan.get("selection_detail", {})
                if sel_detail:
                    with st.expander("推奨理由の詳細", expanded=False):
                        if sel_detail.get("why_bet_type"):
                            st.markdown(f"**券種選択:** {sel_detail['why_bet_type']}")
                        if sel_detail.get("why_combo"):
                            st.markdown(f"**組み合わせ:** {sel_detail['why_combo']}")
                        if sel_detail.get("why_not_other"):
                            st.markdown(f"**他券種比較:** {sel_detail['why_not_other']}")

            # ── 全券種推奨一覧 ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 券種別推奨買い目")
            st.caption("同じ馬の組み合わせを券種ごとに表示します。どれを買うか選択してご利用ください。")
            _all_plans = recommend_all_bet_types(
                result.get("features", []),
                ev_table_ui,
                race_structure,
                bankroll,
                race_pace_ev,
                horse_roles=horse_roles,
            )
            if _all_plans:
                _risk_icon = {"低": "🟢", "中": "🟡", "高": "🔴"}
                for _ap in _all_plans:
                    with st.expander(
                        f"{_risk_icon.get(_ap.get('risk_level','中'), '⚪')} "
                        f"**{_ap['bet_type']}** — {_ap['ticket_count']}点 / ¥{_ap['total_stake']:,}",
                        expanded=(_ap["bet_type"] in ("馬連", "三連複BOX")),
                    ):
                        st.caption(_ap.get("reason", ""))
                        for _t in _ap.get("tickets", []):
                            combo_str = " — ".join(_t.get("combination", []))
                            st.markdown(f"- {combo_str}　¥{_t['stake']:,}")

            # 券種別EV速報
            st.markdown("**券種別 EV（参考）**")
            sum_cols = st.columns(5)
            for ci, bt in enumerate(["単勝", "複勝", "馬連", "ワイド", "3連複"]):
                row_r = best_by_type.get(bt)
                if row_r:
                    sum_cols[ci].metric(bt, f"EV {row_r['ev']:.2f}", delta=f"hit:{row_r['ai_hit_prob']*100:.1f}%")
                else:
                    sum_cols[ci].metric(bt, "-")
```

新しいコード（上記と丸ごと置換）：

```python
        # ── ① AI馬券師の推奨買い目 ─────────────────────────────────────
        st.subheader("🏇 AI馬券師の推奨買い目")
        st.caption("1枚100円固定。AIが全券種を評価し、最適な組み合わせを提示します。")

        _bm_plans = recommend_betmaster_plans(
            result.get("features", []),
            race_structure,
            horse_roles=horse_roles,
            race_pace=race_pace_ev,
        )
        _bm_primary = select_primary_betmaster(_bm_plans, race_structure)

        # ── 主推奨ハイライト ───────────────────────────────────────────
        if _bm_primary:
            st.markdown("#### ★ 特におすすめ")
            _pri_legs = _bm_primary.get("formation_legs") or {}
            _pri_count = _bm_primary.get("ticket_count", 0)
            _pri_budget = _bm_primary.get("budget", 0)

            with st.container(border=True):
                st.markdown(f"**{_bm_primary['bet_type']}**　"
                            f"🎯 {_pri_count}点 × 100円 = **¥{_pri_budget:,}**")
                if _pri_legs:
                    for leg_label, leg_horses in _pri_legs.items():
                        st.markdown(f"&nbsp;&nbsp;**{leg_label}:** {'　'.join(leg_horses)}")
                st.caption(_bm_primary.get("reason", ""))
        else:
            st.warning("🚫 現在の予測精度では自信を持って推奨できる券種がありません。")

        # ── 全券種一覧 ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 全券種の評価")
        _risk_icon = {"最低": "⚪", "低": "🟢", "中": "🟡", "高": "🔴"}
        for _bp in _bm_plans:
            _is_primary = (_bm_primary is not None and _bp["bet_type"] == _bm_primary["bet_type"])
            _icon = _risk_icon.get(_bp.get("risk_level", "中"), "🟡")
            if _bp.get("confidence_ok") and _bp.get("tickets"):
                _cnt   = _bp["ticket_count"]
                _bdgt  = _bp["budget"]
                _label = (f"{_icon} **{_bp['bet_type']}**"
                          + ("　⭐特におすすめ" if _is_primary else "")
                          + f"　{_cnt}点 × 100円 = ¥{_bdgt:,}")
                with st.expander(_label, expanded=_is_primary):
                    _legs = _bp.get("formation_legs") or {}
                    if _legs:
                        for _leg_label, _leg_horses in _legs.items():
                            st.markdown(f"**{_leg_label}:** {'　'.join(_leg_horses)}")
                    st.caption(_bp.get("reason", ""))
                    _tix = _bp.get("tickets", [])
                    if _tix and len(_tix) <= 20:
                        st.dataframe(
                            pd.DataFrame([{
                                "組み合わせ": " - ".join(t.get("combination", [])),
                                "金額": f"¥{t.get('stake', 0):,}",
                            } for t in _tix]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    elif _tix:
                        st.caption(f"（{len(_tix)}点の組み合わせ — 点数が多いため省略表示）")
                        st.dataframe(
                            pd.DataFrame([{
                                "組み合わせ": " - ".join(t.get("combination", [])),
                                "金額": f"¥{t.get('stake', 0):,}",
                            } for t in _tix[:10]]),
                            use_container_width=True,
                            hide_index=True,
                        )
            else:
                _no_reason = _bp.get("no_pick_reason", "")
                st.markdown(f"{_icon} {_bp['bet_type']}　— 今回は選択なし"
                            + (f"（{_no_reason}）" if _no_reason else ""))

        # ── EV速報（オッズ入力時のみ）─────────────────────────────────
        ticket_evs_summary: list = []
        if ev_table_ui:
            ticket_evs_summary = build_ticket_ev_table(
                result.get("features", []),
                race_structure=race_structure,
                ev_table=ev_table_ui,
                forced_axis=None,
                horse_roles=horse_roles,
            )
            best_by_type: dict = {}
            for r in ticket_evs_summary:
                bt = r["bet_type"]
                if bt not in best_by_type or r["ev"] > best_by_type[bt]["ev"]:
                    best_by_type[bt] = r
            if best_by_type:
                st.markdown("---")
                st.markdown("**券種別 EV（参考 — オッズ入力時）**")
                sum_cols = st.columns(5)
                for ci, bt in enumerate(["単勝", "複勝", "馬連", "ワイド", "3連複"]):
                    row_r = best_by_type.get(bt)
                    if row_r:
                        sum_cols[ci].metric(bt, f"EV {row_r['ev']:.2f}",
                                            delta=f"hit:{row_r['ai_hit_prob']*100:.1f}%")
                    else:
                        sum_cols[ci].metric(bt, "-")
```

- [ ] **Step 2: アプリの構文チェック**

```bash
cd /Users/ryokarahashi/keiba_ai && python -m py_compile keiba_app.py && echo "syntax OK"
```

期待: `syntax OK`

- [ ] **Step 3: 全テストを実行**

```bash
cd /Users/ryokarahashi/keiba_ai && python -m pytest tests/test_betmaster.py -v
```

期待: 全テスト PASS

- [ ] **Step 4: コミット**

```bash
cd /Users/ryokarahashi/keiba_ai && git add keiba_app.py && git commit -m "feat: replace betting UI with AI馬券師 full-ticket-type recommendation"
```

---

## 完了確認チェックリスト

- [ ] `python -m pytest tests/test_betmaster.py -v` → 全 PASS
- [ ] `python -m py_compile keiba_app.py` → syntax OK
- [ ] `python -m py_compile value_ai.py` → syntax OK
- [ ] Streamlit を手動起動し、推奨セクションが表示されることを確認
  ```bash
  streamlit run keiba_app.py --server.headless true &
  sleep 3 && curl -s http://localhost:8501 | head -5
  ```

---

## 自己レビュー

### Spec カバレッジ確認
| 要件 | 対応タスク |
|------|---------|
| 全券種（単勝〜三連単）対応 | Task 2 |
| 1枚100円固定 | Task 2（BETMASTER_TICKET_UNIT） + Task 4（bankroll=100） |
| 主推奨ハイライト | Task 3（select_primary_betmaster） + Task 5（UI） |
| フォーメーション表示 | Task 2（formation_legs） + Task 5（UI） |
| AI絞り + 全頭2パターン | Task 2（三連複/三連単） |
| 自信なし時「選択なし」 | Task 2（confidence_ok + no_pick_reason） + Task 5（UI） |
| 軍資金入力コメントアウト | Task 4 |
| 枠連はスコープ外 | — |

### 型整合性確認
- `_bm_plan()` が返す `formation_legs: Optional[Dict[str, List[str]]]` → Task 5 UI で `.items()` を正しく呼んでいる ✓
- `_bm_formation_trio_tickets()` が返す `List[Dict]` の `"combination"` キー → Task 5 UI で `t.get("combination", [])` ✓
- `select_primary_betmaster()` が返す `Optional[Dict]` → Task 5 で `if _bm_primary:` でガード ✓
