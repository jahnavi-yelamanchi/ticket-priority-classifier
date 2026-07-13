# Triage design brief

## Product role

Triage has one job: accept a support ticket and return a priority classification. The landing experience is the product demo, not a separate marketing site or a dashboard.

## First viewport

- Header: `Triage` wordmark at left; `Documentation`, `GitHub`, and a white `Try the API` pill at right.
- Heading: `Know what needs attention.` in large, tightly tracked white display type.
- Supporting copy: one muted sentence explaining that Triage classifies support tickets by priority.
- Primary panel: ticket text area on the left; a post-submission priority result on the right; `Classify priority` is the only primary action.

## System

| Element | Decision |
| --- | --- |
| Canvas | Warm near-black; no light-mode page. |
| Type | Inter or Geist fallback; oversized display with aggressive negative tracking; compact readable body copy. |
| Surfaces | Charcoal elevated panels with subtle hairline borders. |
| Actions | White, rounded pill primary action; charcoal secondary actions. |
| Accent | Electric blue is reserved for links and input focus. |
| Atmosphere | One violet-to-magenta spotlight card below the demo; gradients never fill an entire section. |
| Responsive behavior | Two-column demo collapses to one column below tablet width; controls retain 44px targets. |

## Content guardrails

- The response UI shows real priority, confidence, and probability values from the API only.
- Evaluation and latency values render only after recorded metrics exist.
- Do not claim comparisons with other models, fabricated performance, certifications, or explanatory “AI signals.”
- The visual vocabulary is inspired by the supplied reference only; Framer assets, wordmarks, fonts, and copy are not reused.
