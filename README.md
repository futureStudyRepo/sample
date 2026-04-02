
# 🌐 GitHub Pages로 정적 웹사이트 만들기

**GitHub Pages**는 GitHub 저장소를 기반으로 한 무료 정적 웹사이트 호스팅 서비스입니다.  
사용자/프로젝트/조직 단위로 웹사이트를 만들 수 있으며, `.github.io` 도메인을 기본 제공합니다.

---

## ✅ 방법 1: 기본 HTML, CSS, JS로 정적 웹페이지 만들기

### 📌 특징
- HTML/CSS/JS만 사용한 간단한 정적 사이트
- 별도 설치 없이 바로 배포 가능

### 🔧 작업 절차
1. GitHub에 저장소 생성 (예: `my-page`)
2. `index.html` 파일을 저장소 루트에 추가
3. **Settings > Pages** 이동
4. **Source**를 `main` 브랜치 + 루트 폴더 선택
5. 저장하면 `https://[사용자명].github.io/[저장소명]/` 에서 확인 가능

### 🎨 예시
- [샘플 보기](https://leeapgil.github.io)

---

## ✅ 방법 2: Jekyll 템플릿으로 블로그 스타일 웹페이지 만들기

### 📌 특징
- Markdown 기반 블로그 지원
- 다양한 테마 지원 (`_config.yml`로 설정)
- GitHub Pages가 기본 지원

### 🎨 예시
- [샘플 보기](https://futurestudyrepo.github.io)

### 🌟 대표 테마
- [Minimal Mistakes 테마](https://github.com/mmistakes/minimal-mistakes)

---

### 🖥️ 로컬 테스트 환경 구성 (Windows 기준)

#### 1️⃣ Ruby 설치
- 공식 사이트: [https://rubyinstaller.org/downloads/](https://rubyinstaller.org/downloads/)
- 다운로드 후 실행하여 설치

#### 2️⃣ Jekyll 및 Bundler 설치 (터미널 or CMD)
```bash
gem install jekyll bundler
```

#### 3️⃣ 프로젝트 실행
- 최초 실행```jekyll new . --force```
```bash
bundle exec jekyll serve
```
> `index.md`, `_posts`, `_config.yml` 등이 있는 루트에서 실행

---

### 🛠️ 에러 발생 시 대처

#### 🔁 기본 해결 방법
```bash
gem install jekyll-feed -v 0.1
bundle install
bundle exec jekyll serve
```

---

### 📂 파일 구조 예시 (Jekyll)
```
.
├── _config.yml
├── index.md
├── _posts/
├── _layouts/
├── _includes/
└── assets/
```

---

물론입니다! 🧩 아까 만들어드린 **Jekyll 기본 프로젝트**의 각 파일이 **무슨 역할을 하는지** 아래쪽에 **예시와 함께** 하나하나 쉽게 설명드릴게요.

---

## 📁 `sample/` 프로젝트 설명

### 1. 🛠 `/_config.yml`

Jekyll 전체 사이트 설정 파일입니다.

```yaml
title: My Jekyll Blog
description: Welcome to my simple blog
url: "https://yourusername.github.io"
```

🔍 예시:
- `title`: 블로그 제목 (헤더에 보여짐)
- `url`: GitHub Pages 주소

---

### 2. 📄 `index.md`

블로그의 홈 화면입니다. Markdown 문법으로 글 목록을 보여주는 역할을 합니다.

```markdown
# Welcome to My Blog

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
```

🔍 예시:
- 블로그에 작성된 글들이 **자동으로 리스트**로 나옵니다.

---

### 3. 📁 `_posts/2025-04-24-welcome.md`

블로그 글 하나입니다. **파일명은 날짜 + 제목**으로 되어야 합니다.

```markdown
---
layout: default
title: "Welcome Post"
date: 2025-04-24
---

첫 포스트입니다!
```

🔍 예시:
- `title`: 글 제목
- `layout`: 어떤 HTML 틀을 사용할지 (`default.html` 사용)

---

### 4. 📁 `_layouts/default.html`

모든 페이지가 사용하는 **HTML 레이아웃(틀)** 입니다.

```html
<html>
  <head><title>{{ page.title }}</title></head>
  <body>
    {% include header.html %}
    {{ content }}
  </body>
</html>
```

🔍 예시:
- 모든 글이나 페이지의 공통된 구조 (헤더 + 내용)
- `{{ content }}`에 각 글/페이지의 내용이 들어옴

---

### 5. 📁 `_includes/header.html`

`헤더`로 들어가는 HTML 조각입니다. `default.html`에서 포함됩니다.

```html
<header>
  <h1>{{ site.title }}</h1>
  <p>{{ site.description }}</p>
</header>
```

🔍 예시:
- 블로그 이름과 설명이 항상 위에 보이게 함

---

### 6. 📁 `assets/style.css`

블로그의 스타일을 담당하는 CSS 파일입니다.

```css
body {
  font-family: sans-serif;
  background: #fdfdfd;
  color: #333;
}
```

🔍 예시:
- 배경색, 글자색, 폰트 등 **디자인 변경은 이곳에서!**

---

### 7. 📄 `.gitignore`

Jekyll 빌드시 생성되는 `_site/` 폴더는 GitHub에 올릴 필요 없으니 무시하게 합니다.

```gitignore
_site/
```

---

## 📌 전체 작동 흐름 요약

| 단계 | 설명 |
|------|------|
| 1️⃣ 글을 `_posts`에 작성 | `2025-04-24-welcome.md` |
| 2️⃣ `index.md`에서 글 목록 자동 표시 | `{% for post in site.posts %}` |
| 3️⃣ `default.html`이 레이아웃으로 사용됨 | `{{ content }}` 위치에 글 삽입 |
| 4️⃣ `header.html` 포함되어 사이트 이름 표시 | 상단 고정 |
| 5️⃣ `style.css`로 예쁘게 꾸며짐 | CSS 기반 디자인 |
| 6️⃣ GitHub에 업로드 시 자동 배포 | `_config.yml` 기준 빌드됨 |

---

---
## 레포지토리 페이지 적용 방법

### ✅ 1. GitHub Pages 설정

1. **해당 레포지토리로 이동**  
   → 예: `https://github.com/사용자명/레포지토리명`

2. **Settings 탭 클릭**

3. **좌측 메뉴에서 `Pages` 선택** (`Code and automation` 섹션 아래 있음)

4. **"Source" 항목 확인**  
   → 보통 `Deploy from a branch`로 설정되어 있어야 하고,  
   → `Branch`: `main` (또는 `master`)  
   → `/ (root)` 또는 `/docs` 선택 가능

   🔹 `index.html`이 있는 경로가 `루트`이면 `/ (root)` 선택해야 합니다.

5. **저장 (`Save`) 클릭**

---

### ✅ 2. 파일 위치 

- `index.html`은 반드시 **레포지토리 루트** 또는 `/docs` 디렉토리에 있어야 합니다.
- 파일 이름은 **소문자 `index.html`** 정확히 이렇게!

---

### ✅ 3. URL 확인

- 설정이 완료되면 아래 형식으로 접속합니다:

```txt
https://사용자명.github.io/레포지토리명/
```

예:  
`https://leapgil.github.io/myproject/`

📌 루트 레포지토리 (`leapgil.github.io`)는 레포지토리명 생략 가능!

---

### 🛠️ 4. 반영까지 기다림

- 설정 후 반영되기까지 **1~2분 정도 소요**될 수 있어요.
- 페이지가 안 뜨면 새로고침하거나 시크릿 모드에서 접속해 보세요.

---

### 🚨 오류 상황 점검

- `404` 오류: 파일 위치나 설정 경로 확인 필요
- CSS/JS 안 먹는 경우: 상대경로, 파일 위치 확인

---

# ✅ Jekyll 테마 쉽게 변경하는 방법

## 방법 1. GitHub Pages에서 기본 테마 설정 (가장 쉬움)
1. `_config.yml` 파일 열기  
2. 아래처럼 `theme:` 설정만 바꿔주면 끝!

```yaml
theme: minima   # 기본 테마 이름
```

- 바꿀 수 있는 공식 테마 예시:
  - `minima`
  - `jekyll-theme-cayman`
  - `jekyll-theme-hacker`
  - `jekyll-theme-midnight`
  - `jekyll-theme-slate`
  - `jekyll-theme-modernist`

> 설정 후 GitHub에 푸시하면 자동 적용됨  
> [공식 테마 목록](https://pages.github.com/themes/) 참고

---

## 방법 2. 로컬에서 커스텀 테마 사용

1. 예쁜 테마 다운로드 (예: Minimal Mistakes)
```bash
git clone https://github.com/mmistakes/minimal-mistakes.git my-site
cd my-site
bundle install
```

2. `_config.yml`과 `_posts/` 폴더 수정해서 내 블로그로 맞춤

---

# ✅ 유명한 예쁜 Jekyll 테마 예시 사이트

| 테마 이름 | 미리보기 | 깃허브 링크 |
|-----------|----------|--------------|
| **Minimal Mistakes** | https://mmistakes.github.io/minimal-mistakes/ | https://github.com/mmistakes/minimal-mistakes |
| **Chirpy** | https://chirpy.cotes.page/ | https://github.com/cotes2020/jekyll-theme-chirpy |
| **TeXt** | https://kitian616.github.io/jekyll-TeXt-theme/ | https://github.com/kitian616/jekyll-TeXt-theme |
| **Tale** | https://chesterhow.github.io/tale/ | https://github.com/chesterhow/tale |
| **Lagrange** | https://lenpaul.github.io/Lagrange/ | https://github.com/LeNPaul/Lagrange |

---

## ✅ 예시 적용 (_config.yml)

```yaml
title: 내 블로그 제목
description: 내 블로그 설명
theme: jekyll-theme-cayman    # 테마만 바꿔도 적용됨
```

---

## ✅ 팁: GitHub Pages에서 바로 바꾸는 법

1. GitHub 저장소 > Settings > Pages
2. `Theme Chooser` 클릭
3. 원하는 테마 고르고 저장하면 자동 적용됨

---
