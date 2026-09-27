"""Visual theme: a Gradio theme plus CSS for the custom HTML components (light and dark mode)."""
import gradio as gr

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_shadow="0 1px 2px rgba(15, 23, 42, 0.05)",
    block_title_text_weight="600",
    block_label_text_weight="600",
    button_primary_background_fill="linear-gradient(90deg, *primary_600, *secondary_600)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_500, *secondary_500)",
)

# Gradio scopes `css` to elements inside its content area, so rules for the outer page containers go into
# an unscoped <style> tag in the page head instead.
HEAD = """<style>
gradio-app .gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
@media (max-width: 700px) {  /* phones: let the page shrink to the screen instead of scrolling sideways */
  gradio-app .gradio-container, gradio-app main, gradio-app .wrap, gradio-app .contain { min-width: 0 !important; max-width: 100% !important; }
}
</style>"""

CSS = """
:root {
  --risk-low: #0ca30c; --risk-moderate: #fab219; --risk-high: #ec835a; --risk-very-high: #d03b3b;
  --pushes-up: #e34948; --pushes-down: #2a78d6;
}
.dark { --pushes-up: #e66767; --pushes-down: #3987e5; }

/* Header */
.hero { background: linear-gradient(120deg, #312e81 0%, #4338ca 48%, #0369a1 100%); color: #fff;
  border-radius: 18px; padding: 24px 28px; display: flex; flex-wrap: wrap; gap: 18px;
  align-items: center; justify-content: space-between; }
.hero h1 { color: #fff !important; font-size: 28px; font-weight: 750; letter-spacing: -0.02em; margin: 0 0 6px; }
.hero p { color: rgba(255,255,255,.86) !important; margin: 0; font-size: 15px; max-width: 560px; }
.hero .links { margin-top: 10px; font-size: 13px; color: rgba(255,255,255,.8); }
.hero .links a { color: #fff !important; text-decoration: underline; text-underline-offset: 3px; }
.hero > div { flex: 1 1 300px; min-width: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }
.chip { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.24); border-radius: 12px;
  padding: 8px 14px; min-width: 104px; }
.chip .v { font-size: 20px; font-weight: 700; color: #fff; }
.chip .l { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: rgba(255,255,255,.78); }

/* Layout helpers */
.sticky-col { position: sticky; top: 12px; align-self: flex-start; }
.section-title { font-size: 15px; font-weight: 650; margin: 4px 0 2px; }
.caption { font-size: 12.5px; color: var(--body-text-color-subdued); margin-top: 6px; line-height: 1.45; }
.preset-row button { font-size: 13px !important; }

/* Gauge + verdict */
.gauge { display: flex; flex-direction: column; align-items: center; padding-top: 4px; }
.gauge svg { width: 100%; max-width: 310px; overflow: visible; }
.gauge .arc { animation: sweep .7s ease-out; }
@keyframes sweep { from { stroke-dashoffset: var(--arc-len); } }
.gauge .pct { font-size: 27px; font-weight: 750; fill: var(--body-text-color); }
.gauge .sub { font-size: 8.5px; fill: var(--body-text-color-subdued); }
.verdict { text-align: center; margin-top: 2px; }
.verdict .headline { font-size: 17px; font-weight: 650; margin: 10px 0 4px; }
.verdict .detail { font-size: 13px; color: var(--body-text-color-subdued); }
.badge { display: inline-flex; align-items: center; gap: 7px; padding: 4px 12px; border-radius: 999px;
  font-weight: 650; font-size: 13px; color: var(--body-text-color);
  background: color-mix(in srgb, var(--tier) 18%, transparent); }
.badge .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--tier); }

/* Driver bars (diverging) and plain bars */
.bars { display: flex; flex-direction: column; gap: 8px; }
.bar-row { display: grid; grid-template-columns: minmax(118px, 42%) 1fr 56px; align-items: center; gap: 10px; font-size: 13px; }
.bar-row .name { line-height: 1.25; }
.bar-row .name small { display: block; color: var(--body-text-color-subdued); font-size: 11.5px; }
.bar-row .track { position: relative; height: 12px; }
.bar-row .track.diverging::before { content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px;
  background: var(--border-color-primary); }
.bar-row .bar { position: absolute; top: 0; height: 12px; border-radius: 4px; animation: grow .5s ease-out; }
.bar-row .bar.up { left: 50%; background: var(--pushes-up); transform-origin: left; }
.bar-row .bar.down { right: 50%; background: var(--pushes-down); transform-origin: right; }
.bar-row .bar.plain { left: 0; background: var(--pushes-down); transform-origin: left; }
@keyframes grow { from { transform: scaleX(0); } }
.bar-row .num { text-align: right; font-variant-numeric: tabular-nums; font-size: 12px; color: var(--body-text-color-subdued); }
.legend { display: flex; gap: 16px; font-size: 12px; color: var(--body-text-color-subdued); margin-bottom: 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }

/* Retention scenarios */
.scenario { display: grid; grid-template-columns: 1fr auto auto; gap: 12px; align-items: center; padding: 10px 12px;
  border: 1px solid var(--border-color-primary); border-radius: 12px; margin-bottom: 8px; font-size: 13.5px;
  background: var(--block-background-fill); }
.scenario .p { font-variant-numeric: tabular-nums; color: var(--body-text-color-subdued); }
.delta { font-weight: 650; font-variant-numeric: tabular-nums; padding: 2px 9px; border-radius: 8px; font-size: 13px; }
.delta.down { background: color-mix(in srgb, var(--pushes-down) 16%, transparent); }
.delta.up { background: color-mix(in srgb, var(--pushes-up) 16%, transparent); }

/* KPI cards, confusion matrix, business view */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.kpi { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 14px 16px; background: var(--block-background-fill); }
.kpi .l { font-size: 11.5px; color: var(--body-text-color-subdued); text-transform: uppercase; letter-spacing: .05em; }
.kpi .v { font-size: 26px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
.kpi .s { font-size: 12px; color: var(--body-text-color-subdued); margin-top: 2px; }
.cm { display: grid; grid-template-columns: 92px 1fr 1fr; gap: 6px; font-size: 13px; }
.cm .h { font-size: 12px; color: var(--body-text-color-subdued); display: flex; align-items: center; justify-content: center; text-align: center; }
.cm .cell { border-radius: 12px; padding: 14px 8px; text-align: center; }
.cm .good { background: color-mix(in srgb, var(--pushes-down) 15%, transparent); }
.cm .bad { background: color-mix(in srgb, var(--pushes-up) 13%, transparent); }
.cm .n { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.cm .t { font-size: 11.5px; color: var(--body-text-color-subdued); }
.business { font-size: 14px; line-height: 1.7; }
.business b { font-variant-numeric: tabular-nums; }

@media (max-width: 900px) {
  .sticky-col { position: static; }
  .chips { justify-content: flex-start; }
  .chip { flex: 1 1 110px; min-width: 0; }
  .hero { padding: 18px; }
  .hero h1 { font-size: 22px; }
  .bar-row { grid-template-columns: minmax(96px, 40%) 1fr 48px; }
}
@media (max-width: 700px) {  /* phones: wrap form rows and stack side-by-side panels */
  .form, .row { flex-wrap: wrap !important; min-width: 0 !important; }
  .tabs, .tabitem { min-width: 0 !important; }
  .form > *, .row > * { min-width: min(140px, 100%) !important; flex: 1 1 140px !important; }
  .row > .column { flex: 1 1 100% !important; }  /* stack side-by-side panels */
  .scenario { grid-template-columns: 1fr auto; }
  .scenario .p { grid-column: 1; }
}
"""
