# Survey Tool — 조사설정 엑셀 기반 개편본

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 폰트 적용

프로젝트의 `fonts/` 폴더에 사용할 한글 폰트 파일을 넣습니다.

- 지원 형식: `.ttf`, `.otf`, `.ttc`
- 앱을 재실행하면 PNG 그래프의 폰트 목록에 자동 표시됩니다.
- 고객용 HTML 대시보드 ZIP에도 폰트 파일이 `assets/fonts/`로 함께 포함됩니다.
- 폰트 파일 자체는 포함하지 않았으므로 사용·배포 권한이 있는 파일을 직접 넣어야 합니다.

## 작업 흐름

1. CSV/XLSX/SAV 데이터 파일 업로드
2. 표준 조사설정 XLSX 업로드
3. 통계표 엑셀 추출
4. 전체값 기준 PNG 그래프 일괄 생성
5. 고객용 정적 HTML 대시보드 ZIP 생성

고객용 ZIP의 `index.html`은 Streamlit 없이 동작합니다. 웹서버의 원하는 폴더에 ZIP 내용을 올리면 해당 주소를 고객에게 전달할 수 있습니다.

대시보드 기능:

- 문항 검색
- 문항 선택 드롭다운
- 파트·문항영역 필터
- 이전·다음 문항
- 주요 문항 즐겨찾기
- 비교 기준 선택
- 선택 문항만 인터랙티브 차트로 렌더링
- Plotly 카메라 버튼을 통한 PNG 저장

## 기본 분석 기준

- 비율 기준: 문항별 유효응답
- 척도형 보기: 코드 순서 유지
- 기본 그래프: 세로 막대
- 기본 그래프 생성 대상: 단일응답, 복수응답, 척도형, 순위형
- 복수응답: 기존 그룹 규칙과 선택코드 1 사용
- 가중치: `wt`, `weight`, `가중치`, `표본가중치` 자동 감지

## GitHub에 fonts 폴더 올리기

Git은 빈 폴더를 추적하지 않으므로 `fonts/.gitkeep`을 포함했습니다. `.ttf`, `.otf`, `.ttc`, `.woff`, `.woff2` 파일은 `.gitignore`의 예외 규칙으로 추적되도록 설정되어 있습니다.

1. 사용할 폰트 파일을 `fonts/`에 복사합니다.
2. Windows에서는 `add_fonts_to_git.bat`를 실행합니다.
3. GitHub Desktop에서 변경 파일을 커밋한 뒤 Push합니다.

직접 명령어를 사용할 경우:

```bash
git add .gitignore .gitattributes fonts/.gitkeep
git add -f fonts/*.ttf fonts/*.otf fonts/*.ttc fonts/*.woff fonts/*.woff2
git commit -m "Add project fonts"
git push
```

폰트 파일을 저장소에 포함하기 전에는 해당 폰트의 재배포 및 웹 임베딩 라이선스를 확인해야 합니다.

## v5 추가사항

- `문항설정` 시트에 `문항영역` 열 추가
- 문항영역을 대시보드 필터, 그래프 폴더, PNG ZIP 경로, 통계표 시트에 자동 연결
- 분석 가능한 모든 문항을 그래프·대시보드 기본 포함
- 3단계 설정표는 Streamlit `data_editor`에서 엑셀처럼 직접 수정 가능
- 그래프 설정표는 자주 수정하는 열만 표시하여 문항이 많아도 빠르게 검토 가능
- 선택 시트 `그래프설정` 지원
  - `문항번호`
  - `막대색상`: 예) `#406A9F`
  - `줄바꿈`: 예) `매우 만족=>매우|만족;전혀 만족하지 않음=>전혀 만족하지|않음`
- `고객용 Dashboard 폴더 생성` 버튼으로 `output/<패키지명>/index.html`, `data.json`, `assets/` 생성
- 각 작업 페이지의 안내 캡션 제거
