// Payment Integrity Concept Governance deck — edit CONFIG and THEME, then run: node build_deck.js
const pptxgen = require('pptxgenjs');
const { applyTheme } = require('/mnt/skills/public/pptx/scripts/apply_theme.js');

// ---------- CONFIG: every threshold shown in the deck ----------
const CONFIG = {
  'P-MO-REJ': '70%',
  'P-TIER-ADJ': 'TBD',
  'P-NO-HIGH': '80%',
  'P-NO-LOW': '50% or 60% (decide)',
  'P-OC-NEST': 'TBD',
  'P-HR-MIN': 'TBD',
  'P-SMP': 'TBD',
  'P-OVT-MAX': 'TBD',
  'P-EST-MIN': 'TBD',
  'P-LPQ': 'TBD',
  'P-IAS': 'Equal weights; bands TBD',
  'P-GS-PTS': '100 / 75 / 50 / 25',
  'P-CL': 'TBD',
  'P-POL-SLA': 'TBD',
};

const THEME = {
  name: 'PI Governance',
  headFontFace: 'Cambria',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '1B2229', lt1: 'FFFFFF', dk2: '0F4C5C', lt2: 'EEF3F4',
    accent1: '0F4C5C', accent2: '3E8A99', accent3: 'E36414', accent4: '8C99A3',
    accent5: '2E8B57', accent6: 'B23A48', hlink: '0F4C5C', folHlink: '3E8A99',
  },
};
const BORDER = 'C9D3D8';

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'Payment Integrity Concept Development and Governance Framework';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- Layouts ----------
pres.defineSlideMaster({
  title: 'TITLE_DARK',
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 1.9, w: 11.7, h: 2.0, fontSize: 40, bold: true, color: C.background1, valign: 'bottom' }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 4.15, w: 11.7, h: 1.2, fontSize: 20, color: C.background2, valign: 'top' }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'CONTENT',
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontSize: 30, bold: true, color: C.text2, valign: 'middle' }, text: '' } },
    { text: { text: 'Payment Integrity Concept Governance Framework', options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: 'right' },
});
pres.defineSlideMaster({
  title: 'CLOSING_DARK',
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 0.6, w: 11.7, h: 1.0, fontSize: 36, bold: true, color: C.background1, valign: 'middle' }, text: '' } },
  ],
});

// ---------- Helpers ----------
const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: 'top' }, o));
const box = (s, x, y, w, h, fill, o = {}) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, Object.assign({
  x, y, w, h, rectRadius: 0.08, fill: { color: fill, transparency: o.transparency || 0 },
  line: o.line ? { color: o.line, width: 1 } : { type: 'none' },
}, o.name ? { objectName: o.name } : {}));
const card = (s, x, y, w, h, head, body, o = {}) => {
  box(s, x, y, w, h, o.fill || C.background2, { transparency: o.transparency, name: o.name });
  const pad = 0.22;
  T(s, head, { x: x + pad, y: y + 0.18, w: w - 2 * pad, h: 0.4, fontSize: o.headSize || 16, bold: true, color: o.headColor || C.text2 });
  if (body) T(s, body, { x: x + pad, y: y + 0.62, w: w - 2 * pad, h: h - 0.75, fontSize: o.bodySize || 14, color: o.bodyColor || C.text1 });
};
const table = (s, header, rows, o) => {
  const fs = o.fontSize || 12;
  const border = { type: 'solid', pt: 0.5, color: BORDER };
  const data = [header.map(h => ({ text: h, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: fs } }))];
  rows.forEach((r, i) => data.push(r.map((c, j) => {
    const cell = typeof c === 'object' ? c : { text: String(c) };
    return { text: cell.text, options: Object.assign({ fontSize: fs, color: C.text1, fill: { color: i % 2 ? C.background2 : C.background1 }, bold: j === 0 }, cell.options || {}) };
  })));
  s.addTable(data, { x: o.x, y: o.y, w: o.w, colW: o.colW, border, margin: 0.06, valign: 'middle', rowH: o.rowH });
};
const arrowR = (s, x1, x2, y) => s.addShape(pres.shapes.LINE, { x: x1, y, w: x2 - x1, h: 0, line: { color: C.accent4, width: 1.5, endArrowType: 'triangle' } });
const content = (section, title) => { const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: section }); s.addText(title, { placeholder: 'title' }); return s; };

// ---------- 1. Title ----------
pres.addSection({ title: 'Overview' });
let s = pres.addSlide({ masterName: 'TITLE_DARK', sectionTitle: 'Overview' });
s.addText('Payment Integrity Concept Development & Governance Framework', { placeholder: 'title' });
s.addText('One governed lifecycle for DRG, APC, HBA and future programs', { placeholder: 'body' });

// ---------- 2. Executive summary ----------
s = content('Overview', 'One lifecycle, measured on sustained recovery');
[
  ['01', 'One lifecycle, many programs', 'Gates, tiers, overlap logic, KPIs and RACI are common. DRG, APC and HBA logic plugs in as modules.'],
  ['02', 'Incrementality over volume', 'A concept earns its place by the claims and dollars no model or existing concept already catches.'],
  ['03', 'Sustained findings, not raw hits', 'Appeal and overturn rates are quality metrics from the evidence gate through to LPQ classification.'],
  ['04', 'Configurable thresholds', 'Every cut-off is a named parameter with an owner, a current value and a calibration method.'],
].forEach((c, i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.55 + Math.floor(i / 2) * 2.7;
  box(s, x, y, 5.95, 2.45, C.background2, { name: `summary-${i + 1}` });
  T(s, c[0], { x: x + 0.3, y: y + 0.3, w: 1.0, h: 0.6, fontSize: 28, bold: true, color: C.accent3, fontFace: THEME.headFontFace });
  T(s, c[1], { x: x + 1.3, y: y + 0.35, w: 4.4, h: 0.5, fontSize: 18, bold: true, color: C.text2 });
  T(s, c[2], { x: x + 1.3, y: y + 0.95, w: 4.4, h: 1.3, fontSize: 14 });
});

// ---------- 3. Core vs modules ----------
s = content('Overview', 'Governance is common; program logic plugs in');
table(s, ['Layer', 'Common core (all programs)', 'Program module (DRG, APC, HBA, future)'], [
  ['Concept definition', 'Concept ID, card, owner, version, status', 'Target types: MS-DRG, MCC/CC, PDX, APC, CPT/HCPCS, revenue code'],
  ['Actions', 'Action taxonomy and codes', 'Which actions apply (PDX switch is DRG-only; units correction is outpatient-led)'],
  ['Financial impact', 'Estimation formula and dollar stages', 'Re-grouping or re-pricing logic (grouper, APC pricer, fee schedule)'],
  ['Overlap', 'Model overlap, Jaccard, overlap coefficient, decision matrix', 'Network key (same DRG; same APC family)'],
  ['Guidelines', 'Registry, policy gate, alignment score, change impact', 'ICD-10-CM/PCS guidelines, Coding Clinic, CPT/NCCI, NCD/LCD, payer policy'],
  ['Denials and appeals', 'Taxonomy, finding lifecycle, overturn metrics', 'Program-specific reason codes and appeal routes'],
  ['Tiering and monitoring', 'T1–T3 rules, KPIs, LPQ scoring, clusters', 'Family definitions and performance baselines'],
], { x: 0.6, y: 1.55, w: 12.1, colW: [2.3, 4.5, 5.3], fontSize: 13, rowH: 0.62 });

// ---------- 4. Lifecycle ----------
pres.addSection({ title: 'Lifecycle' });
s = content('Lifecycle', 'Ten stages, eight gates, and a loop for weak performers');
const stages = [
  ['S0', 'G0', 'Intake', 'Concept card and ID'], ['S1', 'G1', 'Classify and tier', 'T1–T3 by specificity'],
  ['S2', 'G2', 'Model overlap', 'Drop model-caught claims'], ['S3', 'G3', 'Network overlap', 'J, OC, containment'],
  ['S4', 'G4', 'Evidence', 'Guideline, alignment, appeals'], ['S5', 'G5', 'SME validation', 'Sample, refine, re-score'],
  ['S6', 'G6', 'Estimation', 'Overturn-adjusted dollars'], ['S7', 'G7', 'POD approval', 'Readiness checklist'],
  ['S8', '', 'Production', 'Monthly scorecard'], ['S9', 'G8', 'Lifecycle decision', 'Retain to deactivate'],
];
const bw = 2.14, gap = 0.35, bh = 1.5;
stages.forEach((st, i) => {
  const col = i % 5, row = Math.floor(i / 5);
  const x = 0.6 + col * (bw + gap), y = 1.6 + row * 2.0;
  const hi = st[0] === 'S7';
  box(s, x, y, bw, bh, hi ? C.accent1 : C.background2, { name: `stage-${st[0]}` });
  T(s, st[1] ? `${st[0]} · ${st[1]}` : st[0], { x: x + 0.18, y: y + 0.15, w: bw - 0.36, h: 0.3, fontSize: 12, bold: true, color: hi ? C.background2 : C.accent3 });
  T(s, st[2], { x: x + 0.18, y: y + 0.45, w: bw - 0.36, h: 0.4, fontSize: 16, bold: true, color: hi ? C.background1 : C.text2 });
  T(s, st[3], { x: x + 0.18, y: y + 0.88, w: bw - 0.36, h: 0.5, fontSize: 12, color: hi ? C.background1 : C.text1 });
  if (col < 4) arrowR(s, x + bw + 0.05, x + bw + gap - 0.05, y + bh / 2);
});
card(s, 0.6, 5.55, 5.95, 1.25, 'Exits', 'Reject at G2 or G3  ·  Return or deactivate at G7  ·  Deactivate or retire at G8', { name: 'exits' });
card(s, 6.75, 5.55, 5.95, 1.25, 'LPQ loop', 'Concepts reclassified as LPQ at G8 re-enter at S0 and pass every gate again, with a version comparison.', { fill: C.accent3, transparency: 85, name: 'lpq-loop' });

// ---------- 5. Gates ----------
s = content('Lifecycle', 'Every gate has explicit entry and exit criteria');
table(s, ['Gate', 'Entry', 'Exit criteria', 'Stage KPI'], [
  ['G0 Intake', 'Request with sponsor', 'Card complete; unique ID; owner and POD named', 'Intake-to-tier days'],
  ['G1 Tier', 'G0 passed', 'Module, action, reason and tier recorded', '% re-tiered later'],
  ['G2 Model overlap', 'Query runs on agreed window', 'Overlap ≤ P-MO-REJ (tier-adjusted); overlap claims removed', 'Model-overlap rejection rate'],
  ['G3 Network overlap', 'Incremental claim set', 'Matrix verdict; duel concluded; LPQ referrals raised', 'Redundancy rejection rate'],
  ['G4 Evidence', 'Proceed or merge verdict', 'Guideline cited for DOS range; alignment not Drifted; overturn history reviewed', '% with full citation'],
  ['G5 SME', 'Evidence record complete', 'Sample ≥ P-SMP; hit-rate lower bound ≥ P-HR-MIN; SME sign-off', 'Refinement success rate'],
  ['G6 Estimation', 'SME-validated version', 'Estimate ≥ P-EST-MIN with confidence level', 'Estimate accuracy'],
  ['G7 Approval', 'Full dossier', 'All RACI sign-offs; readiness checklist; version locked', 'Days to approval'],
  ['G8 Lifecycle', 'Monitoring periods complete', 'Decision logged; LPQs routed back to S0', 'LPQ %, remediation success'],
], { x: 0.6, y: 1.55, w: 12.1, colW: [2.1, 2.6, 4.9, 2.5], fontSize: 12, rowH: 0.5 });

// ---------- 6. Intake ----------
pres.addSection({ title: 'Classification' });
s = content('Classification', 'Six entry pathways feed one concept card and ID');
T(s, 'Entry pathways', { x: 0.6, y: 1.55, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
['Analytics discovery', 'SME-identified', 'Denial and appeal trends', 'Guideline or policy change', 'LPQ remediation', 'Business and strategic'].forEach((p, i) => {
  const x = 0.6 + (i % 2) * 2.8, y = 2.1 + Math.floor(i / 2) * 0.8;
  box(s, x, y, 2.65, 0.62, i === 4 ? C.accent3 : C.background2, { transparency: i === 4 ? 85 : 0, name: `pathway-${i + 1}` });
  T(s, p, { x: x + 0.15, y, w: 2.35, h: 0.62, fontSize: 14, valign: 'middle', color: C.text2, bold: true });
});
T(s, 'Every pathway completes the same mandatory card: objective, targets, action, reason, guideline, expected outcome, owner and POD.', { x: 0.6, y: 4.65, w: 5.4, h: 1.2, fontSize: 14 });
box(s, 6.6, 1.55, 6.1, 1.45, C.text2, { name: 'id-example' });
T(s, 'DRG-871-J9601-DXDROP-CNS-v1', { x: 6.85, y: 1.75, w: 5.6, h: 0.55, fontSize: 22, bold: true, color: C.background1, fontFace: 'Courier New' });
T(s, 'Program · Target · Action · Reason · Version', { x: 6.85, y: 2.4, w: 5.6, h: 0.4, fontSize: 14, color: C.background2 });
table(s, ['Action', 'Meaning', 'Reason', 'Meaning'], [
  ['DXDROP', 'Secondary diagnosis drop', 'CNS', 'Clinically not supported'],
  ['DXSWAP', 'Diagnosis swap', 'DOCINC', 'Documentation inconsistency'],
  ['PDXSW', 'Principal diagnosis switch', 'CODGL', 'Coding guideline violation'],
  ['PXDROP', 'Procedure drop', 'SEQ', 'Sequencing error'],
  ['MODCOR', 'Modifier correction', 'MEDNEC', 'Medical necessity not met'],
  ['UNITCOR', 'Units correction', 'BUNDLE', 'Bundling or unbundling'],
], { x: 6.6, y: 3.25, w: 6.1, colW: [1.15, 1.95, 1.05, 1.95], fontSize: 12, rowH: 0.45 });

// ---------- 7. Tiers ----------
s = content('Classification', 'Broader concepts carry stricter controls');
[
  ['T1', 'Specific code, specific category', 'J96.01 drop on DRG 871'],
  ['T2', 'Code or category family', 'J96.x drop across DRG 870–872'],
  ['T3', 'Broad, multi-code, multi-category', 'Respiratory-failure MCC drop across medical DRGs'],
].forEach((t, i) => {
  const x = 0.6 + i * 4.1;
  box(s, x, 1.55, 3.9, 1.4, C.background2, { name: `tier-${t[0]}` });
  T(s, t[0], { x: x + 0.22, y: 1.7, w: 0.8, h: 0.6, fontSize: 28, bold: true, color: C.accent3, fontFace: THEME.headFontFace });
  T(s, t[1], { x: x + 1.0, y: 1.72, w: 2.7, h: 0.6, fontSize: 14, bold: true, color: C.text2 });
  T(s, t[2], { x: x + 1.0, y: 2.32, w: 2.7, h: 0.55, fontSize: 12 });
});
table(s, ['Control', 'T1', 'T2', 'T3'], [
  ['Model overlap tolerance', 'P-MO-REJ', 'P-MO-REJ − P-TIER-ADJ', 'P-MO-REJ − 2 × P-TIER-ADJ'],
  ['Network overlap', 'Base bands', 'Base bands', 'Tighter high band; OC mandatory'],
  ['SME involvement', 'Coder or SME', 'Coder + clinical SME', '+ second-level review'],
  ['Sample', 'P-SMP', 'Stratified by family', 'Minimum per stratum'],
  ['Approval authority', 'POD lead', 'POD lead + board', 'Board + leadership'],
  ['Monitoring', 'Monthly', 'Monthly', 'Early LPQ trigger'],
], { x: 0.6, y: 3.2, w: 12.1, colW: [2.8, 2.6, 3.0, 3.7], fontSize: 13, rowH: 0.5 });

// ---------- 8. Overlap metrics ----------
pres.addSection({ title: 'Validation' });
s = content('Validation', 'Three overlap metrics catch duplicates, nesting and redundancy');
[
  ['Jaccard (J)', 'J = |A∩B| / |A∪B|', 'How similar are the two claim sets overall?', 'Understates nesting: a small query inside a large one scores low'],
  ['Overlap coefficient (OC)', 'OC = |A∩B| / min(|A|,|B|)', 'Is one query largely inside the other?', 'Symmetric: does not say which one is nested'],
  ['Containment C(A→B)', 'C = |A∩B| / |A|', 'What share of the new concept is already caught?', 'Ignores how much of the existing concept is untouched'],
].forEach((m, i) => {
  const x = 0.6 + i * 4.1;
  box(s, x, 1.55, 3.9, 3.2, C.background2, { name: `metric-${i + 1}` });
  T(s, m[0], { x: x + 0.22, y: 1.72, w: 3.46, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  T(s, m[1], { x: x + 0.22, y: 2.2, w: 3.46, h: 0.5, fontSize: 18, italic: true, color: C.accent3, fontFace: THEME.headFontFace });
  T(s, [{ text: 'Answers: ', options: { bold: true } }, { text: m[2] }], { x: x + 0.22, y: 2.85, w: 3.46, h: 0.8, fontSize: 13 });
  T(s, [{ text: 'Blind spot: ', options: { bold: true } }, { text: m[3] }], { x: x + 0.22, y: 3.7, w: 3.46, h: 0.9, fontSize: 13 });
});
card(s, 0.6, 5.0, 5.95, 1.8, 'Model overlap runs first', 'Share of concept claims and dollars already flagged by green and amber model outputs. Above P-MO-REJ rejects; below, overlap claims are dropped.', { name: 'model-overlap', bodySize: 13 });
card(s, 6.75, 5.0, 5.95, 1.8, 'Why use all three', 'A 1,000-claim query inside a 10,000-claim query scores J = 0.10 but OC = 1.0. J alone would let a nested duplicate through.', { fill: C.accent3, transparency: 85, name: 'why-three', bodySize: 13 });

// ---------- 9. Decision matrix ----------
s = content('Validation', 'Overlap bands route each concept to reject, compare or proceed');
table(s, ['Band', 'Rule', 'Action'], [
  ['High', 'J or C ≥ P-NO-HIGH', 'Reject, or merge as a new version'],
  ['Nested', 'OC ≥ P-OC-NEST, J below high', 'Treat as refinement or re-tier; hit-rate duel'],
  ['Intermediate', 'P-NO-LOW ≤ J < P-NO-HIGH', 'Hit-rate duel'],
  ['Low', 'J < P-NO-LOW and OC < P-OC-NEST', 'Retain original intent; proceed to S4'],
], { x: 0.6, y: 1.55, w: 6.6, colW: [1.5, 2.5, 2.6], fontSize: 13, rowH: 0.75 });
T(s, 'Bands are contiguous, so every overlap value has a route.', { x: 0.6, y: 5.5, w: 6.6, h: 0.5, fontSize: 13, italic: true, color: C.accent4 });
T(s, 'Hit-rate duel: A only, A∩B, B only', { x: 7.6, y: 1.55, w: 5.1, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
T(s, 'Compared on overturn-adjusted hit rate and net recovery per reviewed claim', { x: 7.6, y: 1.95, w: 5.1, h: 0.5, fontSize: 12, color: C.accent4 });
[
  ['Existing is better', 'Remove A∩B from the new concept; proceed on the incremental claims', false],
  ['New is better', 'New keeps full intent; existing concept goes to LPQ remediation', true],
  ['Comparable', 'Merge into one combined version', false],
].forEach((o, i) => {
  const y = 2.6 + i * 1.4;
  card(s, 7.6, y, 5.1, 1.2, o[0], o[1], { fill: o[2] ? C.accent1 : C.background2, headColor: o[2] ? C.background1 : C.text2, bodyColor: o[2] ? C.background1 : C.text1, bodySize: 13, name: `duel-${i + 1}` });
});

// ---------- 10. Intent Alignment Score ----------
s = content('Validation', 'Intent Alignment Score rates how well a concept follows its guideline');
T(s, 'Guideline specificity', { x: 0.6, y: 1.55, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
const pts = CONFIG['P-GS-PTS'].split('/').map(v => v.trim());
[['GS1', 'Explicit coding or reimbursement rule'], ['GS2', 'Authoritative interpretive guidance'], ['GS3', 'Clinical criteria needing judgment'], ['GS4', 'Payer or internal policy only']].forEach((g, i) => {
  const y = 2.1 + i * 0.8, wBar = 2.3 * (4 - i) / 4;
  T(s, g[0], { x: 0.6, y, w: 0.7, h: 0.6, fontSize: 14, bold: true, color: C.text2, valign: 'middle' });
  s.addShape(pres.shapes.RECTANGLE, { x: 1.3, y: y + 0.1, w: wBar, h: 0.4, fill: { color: C.accent2, transparency: i * 18 }, line: { type: 'none' }, objectName: `gs-bar-${i + 1}` });
  T(s, pts[i] || '', { x: 1.3 + wBar + 0.1, y, w: 0.6, h: 0.6, fontSize: 13, bold: true, color: C.accent3, valign: 'middle' });
  T(s, g[1], { x: 4.35, y, w: 2.15, h: 0.6, fontSize: 12, valign: 'middle' });
});
table(s, ['Component', 'Measures'], [
  ['S  Specificity', 'Strength of the guideline basis (GS points)'],
  ['F  Logic fidelity', 'Guideline conditions encoded in the query'],
  ['R  Reason concordance', 'SME findings matching the stated reason'],
  ['D  Appeal defensibility', 'Survival of guideline-related appeals'],
  ['C  Currency', 'Claims inside the guideline effective dates'],
], { x: 6.75, y: 1.55, w: 5.95, colW: [2.3, 3.65], fontSize: 13, rowH: 0.52 });
box(s, 0.6, 5.5, 12.1, 1.3, C.background2, { name: 'ias-formula' });
T(s, 'IAS = wS·S + wF·F + wR·R + wD·D + wC·C', { x: 0.85, y: 5.65, w: 6.0, h: 0.5, fontSize: 20, italic: true, color: C.text2, fontFace: THEME.headFontFace });
T(s, `Weights: ${CONFIG['P-IAS']}`, { x: 0.85, y: 6.2, w: 6.0, h: 0.4, fontSize: 12, color: C.accent4 });
T(s, [{ text: 'Aligned', options: { bold: true, color: C.accent5 } }, { text: ' ≥ P-IAS-HIGH   ' }, { text: 'Partial', options: { bold: true, color: C.accent3 } }, { text: ' between   ' }, { text: 'Drifted', options: { bold: true, color: C.accent6 } }, { text: ' < P-IAS-LOW fails G4' }], { x: 7.0, y: 5.65, w: 5.5, h: 1.0, fontSize: 14, valign: 'middle' });

// ---------- 11. Appeals ----------
s = content('Validation', 'Findings count only when they survive appeal');
['Query hit', 'Validated finding', 'Issued finding', 'Appealed', 'Sustained or overturned', 'Realized recovery'].forEach((st, i) => {
  s.addShape(pres.shapes.CHEVRON, { x: 0.6 + i * 2.02, y: 1.6, w: 2.12, h: 0.95, fill: { color: i === 5 ? C.accent1 : C.accent2, transparency: i === 5 ? 0 : 30 }, line: { type: 'none' }, objectName: `stage-chevron-${i + 1}` });
  T(s, st, { x: 0.95 + i * 2.02, y: 1.6, w: 1.45, h: 0.95, fontSize: 12, bold: true, color: C.background1, valign: 'middle', align: 'center' });
});
[
  ['Appeal rate', 'Appealed ÷ issued findings'],
  ['Overturn rate', 'Overturned ÷ appeals decided'],
  ['Sustain rate', '1 − overturned ÷ issued findings'],
  ['Overturn-adjusted hit rate', 'SME hit rate × sustain rate'],
].forEach((m, i) => card(s, 0.6 + i * 3.06, 2.9, 2.88, 1.45, m[0], m[1], { bodySize: 13, name: `appeal-metric-${i + 1}` }));
card(s, 0.6, 4.6, 12.1, 2.2, 'How appeal intelligence is used', '', { name: 'appeal-uses' });
T(s, [
  { text: 'New concepts: sustained patterns outside existing concepts become intake ideas', options: { bullet: true, breakLine: true } },
  { text: 'Refinement: overturn reasons drive exclusions and tighter logic', options: { bullet: true, breakLine: true } },
  { text: 'LPQ: overturn rate above P-OVT-MAX for consecutive periods triggers review, even with a healthy hit rate', options: { bullet: true, breakLine: true } },
  { text: 'Estimation: expected overturn rate is a direct input', options: { bullet: true } },
], { x: 0.82, y: 5.15, w: 11.6, h: 1.55, fontSize: 14, paraSpaceAfter: 4 });

// ---------- 12. Estimation ----------
s = content('Validation', 'Estimates net out overlap and expected overturns');
box(s, 0.6, 1.55, 12.1, 1.1, C.text2, { name: 'est-formula' });
T(s, 'Est. recovery = N × (1 − MO) × (1 − DUP) × HR × (1 − OVT) × R̄', { x: 0.85, y: 1.55, w: 11.6, h: 1.1, fontSize: 24, italic: true, color: C.background1, valign: 'middle', fontFace: THEME.headFontFace });
table(s, ['Term', 'Meaning'], [
  ['N', 'Eligible annualized claims'], ['MO', 'Model overlap share'], ['DUP', 'Network duplicate share'],
  ['HR', 'SME-validated hit rate'], ['OVT', 'Expected overturn rate'], ['R̄', 'Average recovery per finding'],
], { x: 0.6, y: 2.95, w: 5.6, colW: [1.0, 4.6], fontSize: 13, rowH: 0.48 });
T(s, 'Confidence: High, Medium or Low by sample size and months of comparable history.', { x: 0.6, y: 6.4, w: 5.6, h: 0.45, fontSize: 12, color: C.accent4 });
T(s, 'Keep dollar stages separate', { x: 6.75, y: 2.95, w: 5.95, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
['Estimated', 'Validated', 'Approved', 'Issued', 'Realized (net of overturns)'].forEach((d, i) => {
  const w = 5.95 - i * 0.5;
  box(s, 6.75, 3.45 + i * 0.66, w, 0.54, i === 4 ? C.accent3 : C.accent2, { transparency: i === 4 ? 0 : 70 - i * 10, name: `dollar-stage-${i + 1}` });
  T(s, d, { x: 6.95, y: 3.45 + i * 0.66, w: w - 0.4, h: 0.54, fontSize: 14, bold: true, color: i === 4 ? C.background1 : C.text2, valign: 'middle' });
});

// ---------- 13. Monitoring ----------
pres.addSection({ title: 'Production' });
s = content('Production', 'Concepts become LPQs only after sustained breaches');
table(s, ['Indicator', 'Breach condition'], [
  ['Hit rate', 'Below tier floor'], ['Overturn rate', 'Above P-OVT-MAX'], ['Realization', 'Realized ÷ approved below floor'],
  ['Volume', 'Eligible claims below floor'], ['Effort', 'Recovery per SME hour below floor'], ['Degradation', 'Hit rate drop vs own baseline'],
  ['Redundancy', 'Overlap risen above G2 or G3 limits'],
], { x: 0.6, y: 1.55, w: 6.0, colW: [2.0, 4.0], fontSize: 13, rowH: 0.52 });
T(s, `LPQ thresholds, weights and periods: ${CONFIG['P-LPQ']}`, { x: 0.6, y: 5.95, w: 6.0, h: 0.4, fontSize: 12, color: C.accent4 });
[
  ['Healthy', 'No breaches. Retain.', C.accent5],
  ['Watch', 'Any breach this period. Owner review; no logic change yet.', C.accent3],
  ['LPQ', 'Score above trigger for consecutive periods, or policy invalidation. Back to S0.', C.accent6],
].forEach((h, i) => {
  const y = 1.55 + i * 1.25;
  box(s, 7.0, y, 5.7, 1.1, h[2], { transparency: 85, name: `status-${h[0]}` });
  T(s, h[0], { x: 7.22, y: y + 0.15, w: 1.4, h: 0.8, fontSize: 18, bold: true, color: h[2], valign: 'middle' });
  T(s, h[1], { x: 8.6, y: y + 0.15, w: 3.9, h: 0.8, fontSize: 13, valign: 'middle' });
});
T(s, 'Lifecycle decisions at G8', { x: 7.0, y: 5.4, w: 5.7, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
['Retain', 'Optimize', 'Expand', 'Merge', 'LPQ', 'Deactivate'].forEach((d, i) => {
  const x = 7.0 + (i % 3) * 1.93, y = 5.85 + Math.floor(i / 3) * 0.5;
  box(s, x, y, 1.8, 0.4, C.background2, { name: `decision-${d}` });
  T(s, d, { x, y, w: 1.8, h: 0.4, fontSize: 12, bold: true, color: C.text2, align: 'center', valign: 'middle' });
});

// ---------- 14. Clusters x alignment ----------
s = content('Production', 'Performance clusters and alignment together set the action');
T(s, 'Clustered on hit rate, $ per finding, SME conversion and overturn rate (percentiles within program and tier)', { x: 0.6, y: 1.45, w: 12.1, h: 0.4, fontSize: 13, color: C.accent4 });
[
  ['Star', 'High on all four'], ['Volume driver', 'High hit rate, low $ per finding'], ['High value, fragile', 'High $, high overturn'],
  ['Noisy', 'Low hit rate or conversion'], ['Underperformer', 'Low on most metrics'],
].forEach((c, i) => {
  const y = 2.0 + i * 0.95;
  box(s, 0.6, y, 4.0, 0.8, C.background2, { name: `cluster-${i + 1}` });
  T(s, c[0], { x: 0.8, y: y + 0.08, w: 3.6, h: 0.35, fontSize: 14, bold: true, color: C.text2 });
  T(s, c[1], { x: 0.8, y: y + 0.42, w: 3.6, h: 0.32, fontSize: 12 });
});
const mx = 4.85, my = 2.0, cw = 1.98, ch = 1.35, rh = 1.6;
['Aligned', 'Partially aligned', 'Drifted'].forEach((h, j) => T(s, h, { x: mx + rh + 0.1 + j * (cw + 0.1), y: my, w: cw, h: 0.4, fontSize: 14, bold: true, color: C.text2, align: 'center' }));
const mrows = [
  ['Star / Volume', [['Retain', C.accent5], ['Tighten logic to guideline', C.accent3], ['Compliance review', C.accent6]]],
  ['High value, fragile', [['Improve appeal packets', C.accent3], ['Re-check guideline mapping', C.accent3], ['Suspend pending review', C.accent6]]],
  ['Noisy / Under', [['Logic or ops fix', C.accent3], ['LPQ remediation', C.accent6], ['Deactivate candidate', C.accent6]]],
];
mrows.forEach((r, i) => {
  const y = my + 0.5 + i * (ch + 0.1);
  T(s, r[0], { x: mx, y, w: rh, h: ch, fontSize: 14, bold: true, color: C.text2, valign: 'middle' });
  r[1].forEach((c, j) => {
    const x = mx + rh + 0.1 + j * (cw + 0.1);
    box(s, x, y, cw, ch, c[1], { transparency: 82, name: `matrix-${i + 1}-${j + 1}` });
    T(s, c[0], { x: x + 0.15, y, w: cw - 0.3, h: ch, fontSize: 13, bold: true, align: 'center', valign: 'middle' });
  });
});

// ---------- 15. KPIs ----------
s = content('Production', 'The leadership dashboard tracks quality, not volume');
[
  ['Pipeline efficiency', ['Initiation-to-approval ratio by POD', 'Days to approval', 'Turnaround by stage', 'SME backlog']],
  ['Concept quality', ['Hit rate by tier', 'Overturn-adjusted hit rate', 'Appeal and overturn rate', 'Redundancy rejections']],
  ['Financial effectiveness', ['Estimated to realized $', 'Estimate accuracy', 'Recovery per reviewed claim', 'Recovery per SME hour']],
  ['Sustainability', ['Concepts by health status and cluster', 'LPQ %', 'Remediation success', 'Policy-change impact']],
].forEach((k, i) => {
  const x = 0.6 + i * 3.06;
  box(s, x, 1.55, 2.88, 4.1, C.background2, { name: `kpi-${i + 1}` });
  T(s, k[0], { x: x + 0.2, y: 1.72, w: 2.5, h: 0.65, fontSize: 16, bold: true, color: C.text2 });
  T(s, k[1].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < k[1].length - 1 } })), { x: x + 0.2, y: 2.5, w: 2.5, h: 3.0, fontSize: 13, paraSpaceAfter: 6 });
});
card(s, 0.6, 5.9, 12.1, 0.9, 'Guardrail', '', { fill: C.accent3, transparency: 85, name: 'guardrail' });
T(s, 'Flag any POD whose approvals rise while overturn-adjusted hit rate falls.', { x: 2.2, y: 5.9, w: 10.3, h: 0.9, fontSize: 14, valign: 'middle' });

// ---------- 16. RACI ----------
s = content('Production', 'One accountable owner at every stage');
table(s, ['Stage', 'Analytics', 'SME', 'Policy', 'POD lead', 'Finance', 'Recovery ops', 'Board'], [
  ['S0 Intake', 'R', 'C', 'C', 'A', 'C', 'C', 'I'],
  ['S1 Tier', 'R', 'C', 'I', 'A', 'I', 'I', 'I'],
  ['S2 Model overlap', 'R, A', 'I', 'I', 'C', 'I', 'I', 'I'],
  ['S3 Network overlap', 'R, A', 'C', 'I', 'C', 'I', 'C', 'I'],
  ['S4 Evidence', 'C', 'R', 'R, A', 'C', 'I', 'C', 'I'],
  ['S5 SME validation', 'C', 'R, A', 'C', 'C', 'I', 'I', 'I'],
  ['S6 Estimation', 'R', 'C', 'I', 'C', 'A', 'C', 'I'],
  ['S7 Approval', 'C', 'C', 'C', 'R, A (T1)', 'C', 'C', 'A (T2, T3)'],
  ['S8 Monitoring', 'R', 'I', 'C', 'A', 'C', 'R', 'I'],
  ['S9 Lifecycle', 'R', 'C', 'C', 'A (T1)', 'C', 'C', 'A (T2, T3)'],
  ['Parameter changes', 'R', 'C', 'C', 'C', 'C', 'C', 'A'],
].map(r => r.map((c, j) => j === 0 ? c : { text: c, options: { align: 'center' } })), { x: 0.6, y: 1.55, w: 12.1, colW: [2.5, 1.3, 1.2, 1.2, 1.5, 1.2, 1.5, 1.7], fontSize: 12, rowH: 0.42 });

// ---------- 17. Parameter register ----------
pres.addSection({ title: 'Configuration and roadmap' });
s = content('Configuration and roadmap', 'Every threshold is a configurable parameter');
const val = v => ({ text: v, options: { bold: true, color: C.accent3 } });
table(s, ['ID', 'Parameter', 'Used at', 'Current value'], [
  ['P-MO-REJ', 'Model overlap reject threshold', 'G2', val(CONFIG['P-MO-REJ'])],
  ['P-TIER-ADJ', 'Tier tightening step', 'G2, G3', val(CONFIG['P-TIER-ADJ'])],
  ['P-NO-HIGH', 'Network high-overlap band', 'G3', val(CONFIG['P-NO-HIGH'])],
  ['P-NO-LOW', 'Network low-overlap band', 'G3', val(CONFIG['P-NO-LOW'])],
  ['P-OC-NEST', 'Overlap coefficient for nesting', 'G3', val(CONFIG['P-OC-NEST'])],
  ['P-HR-MIN', 'Minimum validated hit rate', 'G5', val(CONFIG['P-HR-MIN'])],
  ['P-SMP', 'Sample margin, confidence, minimum', 'G5', val(CONFIG['P-SMP'])],
  ['P-OVT-MAX', 'Maximum overturn rate', 'G4, monitoring', val(CONFIG['P-OVT-MAX'])],
  ['P-EST-MIN', 'Minimum annual estimate', 'G6', val(CONFIG['P-EST-MIN'])],
  ['P-LPQ-*', 'LPQ thresholds, weights, periods', 'Monitoring', val(CONFIG['P-LPQ'])],
  ['P-IAS-*', 'Alignment weights and bands', 'G4', val(CONFIG['P-IAS'])],
  ['P-GS-PTS', 'Specificity points GS1–GS4', 'G4', val(CONFIG['P-GS-PTS'])],
  ['P-CL-*', 'Clustering minimum and cut', 'Monitoring', val(CONFIG['P-CL'])],
  ['P-POL-SLA', 'Policy impact assessment days', 'Guidelines', val(CONFIG['P-POL-SLA'])],
], { x: 0.6, y: 1.45, w: 12.1, colW: [1.9, 4.6, 2.2, 3.4], fontSize: 12, rowH: 0.33 });
T(s, 'Edit the Current value column directly. Calibrate quarterly by back-testing overlap and performance values against realized, overturn-adjusted recovery.', { x: 0.6, y: 6.45, w: 12.1, h: 0.45, fontSize: 12, color: C.accent4 });

// ---------- 18. Roadmap ----------
s = content('Configuration and roadmap', 'Foundation first, intelligence later');
[
  ['Foundation', '0–3 months', ['Concept and parameter registries', 'VS Code–Databricks integration with Git', 'Automated overlap notebooks', 'Concept card and gate checklists']],
  ['Scale', '3–6 months', ['Multi-concept data pull and estimates', 'Appeal data joined to concept IDs', 'Scorecards, LPQ scoring, clusters', 'Pod-wise initiation-to-approval view']],
  ['Intelligence', '6–12 months', ['Guideline registry with DOS versions', 'Alignment score automation', 'NLP on overturn narratives', 'Concept graph and embeddings']],
  ['Enterprise', '12 months +', ['Onboard APC, HBA, future modules', 'Quarterly recalibration', 'Estimate-accuracy feedback']],
].forEach((p, i) => {
  const x = 0.6 + i * 3.06;
  box(s, x, 1.55, 2.88, 1.05, C.accent1, { transparency: i * 20, name: `phase-${i + 1}` });
  T(s, p[0], { x: x + 0.2, y: 1.65, w: 2.5, h: 0.45, fontSize: 18, bold: true, color: i < 2 ? C.background1 : C.text2 });
  T(s, p[1], { x: x + 0.2, y: 2.1, w: 2.5, h: 0.35, fontSize: 13, color: i < 2 ? C.background2 : C.text2 });
  box(s, x, 2.75, 2.88, 4.05, C.background2, { name: `phase-body-${i + 1}` });
  T(s, p[2].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < p[2].length - 1 } })), { x: x + 0.2, y: 2.95, w: 2.5, h: 3.7, fontSize: 13, paraSpaceAfter: 8 });
});

// ---------- 19. Decisions ----------
s = pres.addSlide({ masterName: 'CLOSING_DARK', sectionTitle: 'Configuration and roadmap' });
s.addText('Decisions needed to launch', { placeholder: 'title' });
[
  ['1', 'Confirm tier definitions for each program module'],
  ['2', 'Set P-NO-LOW to close the 50–60% overlap gap'],
  ['3', 'Agree the appeal data source and its join key to concept ID'],
  ['4', 'Name owners for the concept, guideline and appeal registries'],
].forEach((d, i) => {
  const y = 2.1 + i * 1.15;
  s.addShape(pres.shapes.OVAL, { x: 0.8, y, w: 0.75, h: 0.75, fill: { color: C.accent3 }, line: { type: 'none' }, objectName: `decision-num-${d[0]}` });
  T(s, d[0], { x: 0.8, y, w: 0.75, h: 0.75, fontSize: 22, bold: true, color: C.background1, align: 'center', valign: 'middle' });
  T(s, d[1], { x: 1.85, y, w: 10.5, h: 0.75, fontSize: 20, color: C.background1, valign: 'middle' });
});

(async () => {
  const out = 'PI_Concept_Governance_Framework.pptx';
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log('written', out);
})();
