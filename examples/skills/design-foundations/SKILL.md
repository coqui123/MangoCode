---
name: design-foundations
description: Universal design principles for color, typography, and layout — dependency of all visual output skills
triggers:
  - presentation
  - slides
  - document
  - pdf
  - visual
---

# Design Foundations

This skill is auto-loaded as a dependency by visual output skills (`pptx`, `docx`, etc.).
It establishes baseline principles that all generated documents should follow.

## Typography

- **Headings**: Use a clean sans-serif (Inter, Helvetica Neue, or system-ui)
- **Body**: 16px / 1rem minimum for readability; 1.5× line-height
- **Code**: Monospace (JetBrains Mono, Fira Code, or Cascadia Code)
- **Hierarchy**: Limit to 3 heading levels per document; use weight, not just size
- **Line length**: 60–80 characters for prose; never full-width on wide viewports

## Color

- **Contrast**: All text must meet WCAG AA (4.5:1 for normal text, 3:1 for large text)
- **Palette**: Use a maximum of 5 colours per document; establish one accent, one text, one background
- **Dark text on light**: default; dark-mode variants are opt-in
- **Avoid**: pure `#000000` black on `#ffffff` white — use near-black/near-white for softer feel

## Layout

- **Whitespace**: Generous — padding ≥ 24px on containers, ≥ 16px between sections
- **Alignment**: Left-align body text; centered headings on title slides only
- **Grid**: 12-column where applicable; maintain consistent gutters
- **Images**: Always include alt-text descriptions

## Slide-Specific Rules (when generating presentations)

- **One idea per slide**: Do not put multiple distinct concepts on one slide
- **Title slide**: Company/project name, presentation title, date
- **Bullets**: ≤ 5 bullets per slide; ≤ 8 words per bullet
- **Fonts on slides**: Minimum 24pt body, 36pt headings — legible from distance
- **Animations**: None unless explicitly requested — keep it professional and print-safe
