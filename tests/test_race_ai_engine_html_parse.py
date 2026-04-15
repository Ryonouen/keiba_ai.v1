import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from race_ai_engine import _extract_horse_rows_from_html


def test_extract_horse_rows_from_html_includes_link_and_numbers():
    html = """
    <table class="Shutuba_Table">
      <tbody>
        <tr>
          <td class="Waku">3</td>
          <td class="Umaban">5</td>
          <td class="HorseName"><a href="https://db.netkeiba.com/horse/2023100001/">テストホース</a></td>
          <td class="Jockey">武豊</td>
          <td class="Kinryo">55.0</td>
          <td class="Odds">4.8</td>
          <td class="Ninki">2</td>
          <td class="Barei">牝3</td>
        </tr>
      </tbody>
    </table>
    """

    rows = _extract_horse_rows_from_html(html)

    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "テストホース"
    assert row["link"] == "https://db.netkeiba.com/horse/2023100001/"
    assert row["gate"] == 3
    assert row["number"] == 5
    assert row["jockey"] == "武豊"
    assert row["win_odds_scraped"] == 4.8
    assert row["popularity"] == 2


def test_extract_horse_rows_no_tbody():
    """tbody なしで tr が table 直下に置かれる場合（皐月賞等の未来レース実ページ構造）でも
    馬を正しく取得できることを確認する。修正前は 0 件になっていた。"""
    html = """
    <table class="Shutuba_Table">
      <thead>
        <tr class="Header">
          <th class="Waku">枠</th>
          <th class="Umaban">馬番</th>
          <th>馬名</th>
        </tr>
      </thead>
      <tr class="HorseList" id="tr_1">
        <td class="Waku Txt_C"><span></span></td>
        <td class="Umaban Txt_C">1</td>
        <td class="HorseInfo">
          <span class="HorseName">
            <a href="https://db.netkeiba.com/horse/2023100001/" target="_blank">テストホースA</a>
          </span>
        </td>
        <td class="Jockey">ルメール</td>
        <td class="Kinryo">57.0</td>
      </tr>
      <tr class="HorseList" id="tr_2">
        <td class="Waku Txt_C"><span></span></td>
        <td class="Umaban Txt_C">2</td>
        <td class="HorseInfo">
          <span class="HorseName">
            <a href="https://db.netkeiba.com/horse/2023100002/" target="_blank">テストホースB</a>
          </span>
        </td>
        <td class="Jockey">川田</td>
        <td class="Kinryo">57.0</td>
      </tr>
    </table>
    """

    rows = _extract_horse_rows_from_html(html)

    assert len(rows) == 2, f"tbody なし構造で 2 頭取得できること (got {len(rows)})"
    names = [r["name"] for r in rows]
    assert "テストホースA" in names
    assert "テストホースB" in names
    assert all(r["link"] for r in rows), "全馬に link が付いていること"
