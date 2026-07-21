import re
import pandas as pd


# --------------------------------------------------
# 1. 보기 번호 매핑
# --------------------------------------------------
CIRCLED_NUM_MAP = {
    "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
    "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10,
    "⑪": 11, "⑫": 12, "⑬": 13, "⑭": 14, "⑮": 15,
    "⑯": 16, "⑰": 17, "⑱": 18, "⑲": 19, "⑳": 20,
    "㉠": 1, "㉡": 2, "㉢": 3, "㉣": 4, "㉤": 5, 
    "㉥": 6, "㉦": 7, "㉧": 8, "㉨": 9, "㉩": 10,
    "㉪": 11, "㉫": 12, "㉬": 13, "㉭": 14,
}
CIRCLED_NUM_SET = set(CIRCLED_NUM_MAP.keys())


# --------------------------------------------------
# 2. 문항 패턴
#    문항은 반드시 알파벳(Q/SQ/DQ/PQ/BQ/CQ)으로 시작한다고 가정
# --------------------------------------------------
ALPHA_QID_PATTERN = re.compile(
    r"^(?P<qid>(?:SQ|DQ|PQ|BQ|CQ|Q)\d+(?:-\d+)*)\.\s*(?P<body>.*)$",
    re.IGNORECASE,
)

# 보기 패턴
OPTION_PATTERNS = [
    ("circled", re.compile(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉠㉡㉢㉣㉤㉥㉦㉧㉨㉩㉪㉫㉬㉭])")),
    ("paren", re.compile(r"(\(\d+\))")),
    ("dot", re.compile(r"(?<![A-Za-z])(\d+\.)")),
    ("plain", re.compile(r"(?<![A-Za-z])(\d+)\s")),  # 필요시 fallback
]


# --------------------------------------------------
# 3. 무시할 줄
# --------------------------------------------------
def is_closing_text(line: str) -> bool:
    s = str(line).strip()
    if not s:
        return False

    patterns = [
        r"설문에\s*응답해\s*주셔서\s*감사합니다",
        r"조사에\s*응답해\s*주셔서\s*감사합니다",
        r"조사에\s*참여해\s*주셔서\s*감사합니다",
        r"응답해\s*주셔서\s*감사합니다",
        r"감사합니다\.?$",
        r"여론조사기관.*였습니다",
        r"에스티아이였습니다",
        r"좋은\s*하루\s*되십시오",
    ]
    return any(re.search(p, s) for p in patterns)


def is_instruction_text(line: str) -> bool:
    s = str(line).strip()
    if not s:
        return False

    patterns = [
        r"^\[.*\]$",           # [학생 정보]
        r"^\[.*\]\s*:.*$",     # [학생 정보] : 강사가 일괄 기입
        r"^※",
        r"^▶",
        r"^■",
        r"^□",
        r"^작성\s*예시",
        r"^단위\s*:",
        r"^주의\s*:",
        r"^안내\s*:",
    ]
    return any(re.search(p, s) for p in patterns)


def is_noise_line(line: str) -> bool:
    s = str(line).strip()
    if not s:
        return True

    if is_closing_text(s) or is_instruction_text(s):
        return True

    if s in {"개", "명", "회", "번"}:
        return True

    return False


# --------------------------------------------------
# 4. 괄호 안/밖 분리 보조
#    괄호 안 로직은 보호하고, 괄호 밖 보기만 분리
# --------------------------------------------------
def split_outside_parentheses(line: str, token_set: set[str]) -> list[str]:
    line = str(line).strip()
    if not line:
        return []

    parts = []
    buf = []
    depth = 0
    started = False

    for ch in line:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)

        is_token = (ch in token_set and depth == 0)

        if is_token:
            if not started:
                started = True
                buf = [ch]
                continue

            prev = "".join(buf).strip()
            if prev:
                parts.append(prev)
            buf = [ch]
            continue

        if started:
            buf.append(ch)
        else:
            buf.append(ch)

    final = "".join(buf).strip()
    if final:
        parts.append(final)

    return parts


# --------------------------------------------------
# 5. 한 줄에 여러 보기 자동 분리
#    괄호 안 로직은 최대한 보존
# --------------------------------------------------
def split_multiple_options(line: str) -> list[str]:
    line = str(line).strip()
    if not line:
        return []

    # 괄호 안 로테이션 같은 설명은 손대지 않음
    if "로테이션" in line:
        # 다만 보기 줄이 실제로 섞인 경우는 보통 드물다고 보고 그대로 둠
        # 문항 라벨에 포함되어도 괜찮다는 운영 원칙
        pass

    # 1) 동그라미 숫자 분리
    if any(ch in line for ch in CIRCLED_NUM_SET):
        split_lines = split_outside_parentheses(line, CIRCLED_NUM_SET)
        if len(split_lines) > 1:
            return split_lines

    # 2) (1) 형태 분리
    if re.search(r"\(\d+\)", line):
        parts = re.split(r"(\(\d+\))", line)
        out = []
        for i in range(1, len(parts), 2):
            token = parts[i]
            rest = parts[i + 1].strip() if i + 1 < len(parts) else ""
            merged = f"{token} {rest}".strip()
            if merged:
                out.append(merged)
        if out:
            return out

    # 3) 1. 형태 분리
    dot_matches = list(re.finditer(r"(?<![A-Za-z])\d+\.", line))
    if len(dot_matches) >= 2:
        out = []
        for idx, m in enumerate(dot_matches):
            start = m.start()
            end = dot_matches[idx + 1].start() if idx + 1 < len(dot_matches) else len(line)
            chunk = line[start:end].strip()
            if chunk:
                out.append(chunk)
        if out:
            return out

    return [line]




# --------------------------------------------------
# 6. 텍스트 정리
# --------------------------------------------------
def normalize_questionnaire_text(text: str) -> list[str]:
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    normalized_lines = []

    for raw_line in text.split("\n"):
        raw_line = re.sub(r"\s+", " ", raw_line).strip()
        if not raw_line:
            continue

        split_lines = split_multiple_options(raw_line)

        for line in split_lines:
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue

            normalized_lines.append(line)

    return normalized_lines


# --------------------------------------------------
# 7. 변수명 / 라벨 정리
# --------------------------------------------------
def normalize_var_name(raw_qno: str) -> str:
    raw_qno = str(raw_qno).strip().upper()

    if re.fullmatch(r"(?:SQ|DQ|PQ|BQ|CQ|Q)\d+(?:-\d+)*", raw_qno, flags=re.IGNORECASE):
        return raw_qno.upper()

    return raw_qno.upper()


def clean_question_label(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_option_label(text: str) -> str:
    text = str(text).strip()
    # 괄호 안 로직은 유지하되, 설문 종료용 안내는 제거
    text = re.sub(r"\(☞[^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -·\n\t")
    return text


# --------------------------------------------------
# 8. 라인 판별
# --------------------------------------------------
def is_alpha_question_start(line: str, question_style: str):
    if question_style in ["auto", "alphabet"]:
        m = ALPHA_QID_PATTERN.match(line)
        if m:
            return {"qid": m.group("qid"), "body": m.group("body").strip()}
    
    if question_style in ["auto", "numeric", "dot", "paren", "bracket"]:
        # (1) 문항, (8)-1 문항, 1. 문항 등 숫자/괄호 혼합형 매칭 시도
        num_pattern = re.compile(r"^(?P<raw_qid>(?:\()?\d+(?:\))?(?:-\d+)*)[.)\]]?\s*(?P<body>.*)$")
        m = num_pattern.match(line)
        if m:
            body = m.group("body").strip()
            raw_qid = m.group("raw_qid")
            clean_qid = raw_qid.replace('(', '').replace(')', '')
            
            # 숫자로 시작하더라도 보기와 구별하기 위해 패턴 확인
            is_definitely_question = False
            # 1. 문항 번호에 하이픈이 포함된 경우 (예: 1-1, (8)-1)
            if "-" in raw_qid:
                is_definitely_question = True
            # 2. 문항의 전형적인 종결어미나 특수 기호 포함
            elif any(x in body for x in ["?", "시오", "입니까", "인가요", "있는지", "이름은", "사유는", "무엇"]):
                is_definitely_question = True
            # 3. 텍스트 길이가 충분히 김 (보통 보기는 짧음)
            elif len(body) >= 30: # 30자 이상이면 문항으로 간주
                is_definitely_question = True
            
            if is_definitely_question:
                return {"qid": f"Q{clean_qid}", "body": body}
                
    return None


def is_option_line(line: str, option_style: str):
    # 1. 1) (1) 형태 확인 (단, '1순위'처럼 구분자 없는 단어는 보기로 잡히지 않도록 제한)
    m = re.match(r"^(?:\()?(?P<num>\d+)(?:[.)\]]+|\s+)\s*(?P<body>.*)$", line)
    if not m:
        # 동그라미 숫자 및 한글 자음 형태 확인
        m2 = re.match(r"^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉠㉡㉢㉣㉤㉥㉦㉧㉨㉩㉪㉫㉬㉭])\s*(.*)$", line)
        if m2:
            circled = m2.group(1)
            return {
                "code": CIRCLED_NUM_MAP[circled],
                "body": m2.group(2).strip(),
            }
        return None
        
    return {
        "code": int(m.group("num")),
        "body": m.group("body").strip(),
    }


# --------------------------------------------------
# 9. 블록 파싱
# --------------------------------------------------
def parse_questionnaire_blocks(
    text: str,
    question_style: str = "auto",
    option_style: str = "auto",
    ignore_instruction_lines: bool = True,
    ignore_trailing_text_after_options: bool = True,
    keep_parenthetical_logic: bool = True
):
    lines = normalize_questionnaire_text(text)

    blocks = []
    current = None

    def flush_current():
        nonlocal current
        if current:
            current["question_lines"] = [x for x in current["question_lines"] if str(x).strip()]
            current["options"] = [x for x in current["options"] if str(x.get("label", "")).strip()]
            blocks.append(current)
            current = None

    for line in lines:
        if ignore_instruction_lines and is_instruction_text(line):
            continue
        if ignore_trailing_text_after_options and is_closing_text(line):
            continue
            
        if is_noise_line(line):
            continue

        # 1) 문항 시작
        alpha_q = is_alpha_question_start(line, question_style)
        if alpha_q:
            flush_current()
            current = {
                "raw_qno": alpha_q["qid"],
                "question_lines": [alpha_q["body"]] if alpha_q["body"] else [],
                "options": [],
            }
            continue

        # 2) 보기
        opt = is_option_line(line, option_style)
        if opt and current is not None:
            current["options"].append({
                "code": opt["code"],
                "label": clean_option_label(opt["body"]) if keep_parenthetical_logic else re.sub(r"\(.*?\)", "", clean_option_label(opt["body"])).strip(),
            })
            continue

        # 3) 번호 없는 줄
        if current is not None:
            if not current["options"]:
                if not is_noise_line(line):
                    current["question_lines"].append(line)
            else:
                continue

    flush_current()
    return blocks


# --------------------------------------------------
# 10. 블록 -> 문항 구조
# --------------------------------------------------
def parse_one_question_block(block: dict):
    var_name = normalize_var_name(block["raw_qno"])
    q_label = clean_question_label(" ".join(block.get("question_lines", [])))
    options = []

    for opt in block.get("options", []):
        label = clean_option_label(opt.get("label", ""))
        if not label:
            continue
        options.append({
            "code": opt["code"],
            "label": label,
        })

    return {
        "var_name": var_name,
        "question_label": q_label,
        "options": options,
    }


# --------------------------------------------------
# 11. 최종 Column Guide DataFrame 생성
# --------------------------------------------------
def questionnaire_text_to_guide_df(
    text: str,
    question_style: str = "auto",
    option_style: str = "auto",
    ignore_instruction_lines: bool = True,
    ignore_trailing_text_after_options: bool = True,
    keep_parenthetical_logic: bool = True
) -> pd.DataFrame:
    blocks = parse_questionnaire_blocks(
        text, 
        question_style, 
        option_style, 
        ignore_instruction_lines, 
        ignore_trailing_text_after_options, 
        keep_parenthetical_logic
    )

    rows = []
    for block in blocks:
        parsed = parse_one_question_block(block)
        var_name = parsed["var_name"]
        q_label = parsed["question_label"]
        options = parsed["options"]

        rows.append({
            "문항/보기번호": var_name,
            "내용": q_label,
            "VALUE LABELS": f"/{var_name}",
        })

        for opt in options:
            rows.append({
                "문항/보기번호": opt["code"],
                "내용": opt["label"],
                "VALUE LABELS": f"{opt['code']} '{opt['label']}'",
            })

    return pd.DataFrame(rows, columns=["문항/보기번호", "내용", "VALUE LABELS"])
