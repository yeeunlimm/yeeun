"""Render report.md as a self-contained HTML file, including inline chart images."""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "report.md"
OUTPUT = ROOT / "seoul-popup-report-2026-10-01.html"


def inline(text: str) -> str:
    text = html.escape(text, quote=True)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
    return text


def image_html(alt: str, relative_path: str) -> str:
    path = (ROOT / relative_path).resolve()
    if not path.is_file() or ROOT not in path.parents:
        raise FileNotFoundError(f"리포트 그림을 찾을 수 없습니다: {relative_path}")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt, quote=True)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def render_markdown(source: str) -> str:
    lines = source.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    list_kind: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            out.append(f"</{list_kind}>")
            list_kind = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush_paragraph(); close_list(); i += 1; continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if image:
            flush_paragraph(); close_list(); out.append(image_html(*image.groups())); i += 1; continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"\|?\s*:?-{3,}", lines[i + 1]):
            flush_paragraph(); close_list()
            headers = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("<div class=\"table-wrap\"><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in headers) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table></div>")
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush_paragraph(); close_list()
            n = len(heading.group(1)); out.append(f"<h{n}>{inline(heading.group(2))}</h{n}>")
            i += 1; continue
        bullet = re.match(r"^\s*-\s+(.*)$", line)
        if bullet:
            flush_paragraph()
            if list_kind != "ul": close_list(); out.append("<ul>"); list_kind = "ul"
            out.append(f"<li>{inline(bullet.group(1))}</li>")
            i += 1; continue
        paragraph.append(line.strip())
        i += 1
    flush_paragraph(); close_list()
    return "\n".join(out)


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {OUTPUT}")
    body = render_markdown(SOURCE.read_text(encoding="utf-8"))
    document = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>서울 야외 팝업 후보·일정 1차 리포트</title>
<style>
:root{{color-scheme:light;--ink:#17212b;--muted:#596273;--line:#e4e9ef;--blue:#1b4965;--coral:#e07a5f;--paper:#fff;--wash:#f5f7f9}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif;line-height:1.75}}
main{{max-width:1040px;margin:40px auto;padding:54px 64px;background:var(--paper);box-shadow:0 12px 40px #18263812;border-radius:16px}}
h1{{font-size:2.2rem;line-height:1.3;letter-spacing:-.04em;margin:0 0 20px}}h2{{font-size:1.45rem;margin:2.4em 0 .7em;padding-top:.55em;border-top:1px solid var(--line);letter-spacing:-.025em}}
p{{margin:.55em 0 1em}}strong{{font-weight:700}}a{{color:#14628b;text-underline-offset:3px}}code{{background:#f0f3f6;padding:.1em .35em;border-radius:4px}}
figure{{margin:1.7em 0 2em}}figure img{{display:block;width:100%;height:auto;border-radius:8px}}figcaption{{font-size:.84rem;color:var(--muted);margin-top:.4em}}
.table-wrap{{overflow-x:auto;margin:1.1em 0 1.8em}}table{{border-collapse:collapse;width:100%;font-size:.94rem}}th,td{{text-align:left;padding:11px 13px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{background:#f3f6f8;font-weight:700}}tbody tr:first-child td{{font-weight:650}}
ul{{padding-left:1.3em}}li{{margin:.35em 0}}h1+p{{color:var(--muted)}}h1+p strong{{color:var(--ink)}}
@media(max-width:700px){{main{{margin:0;padding:30px 20px;border-radius:0}}h1{{font-size:1.7rem}}h2{{font-size:1.25rem}}}}
@media print{{body{{background:#fff}}main{{max-width:none;margin:0;padding:15mm;box-shadow:none}}h2,figure,table{{break-inside:avoid}}}}
</style></head><body><main>{body}</main></body></html>'''
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"생성 완료: {OUTPUT}")


if __name__ == "__main__":
    main()
