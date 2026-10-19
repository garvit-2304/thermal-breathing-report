"""Build the accessible project status report, with fixed presentation pages."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
navy = colors.HexColor('#143B4A')
teal = colors.HexColor('#147D86')
muted = colors.HexColor('#526674')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleA', fontName='Helvetica-Bold', fontSize=30, leading=35, textColor=navy, spaceAfter=18))
styles.add(ParagraphStyle(name='HeadA', fontName='Helvetica-Bold', fontSize=23, leading=28, textColor=navy, spaceAfter=19))
styles.add(ParagraphStyle(name='SubA', fontName='Helvetica-Bold', fontSize=13, leading=17, textColor=teal, spaceBefore=13, spaceAfter=6))
styles.add(ParagraphStyle(name='BodyA', fontName='Helvetica', fontSize=11.2, leading=16, textColor=navy, spaceAfter=10))
styles.add(ParagraphStyle(name='SmallA', fontName='Helvetica', fontSize=9, leading=12.5, textColor=muted, spaceAfter=8))
styles.add(ParagraphStyle(name='CallA', fontName='Helvetica-Bold', fontSize=14, leading=20, textColor=navy, backColor=colors.HexColor('#E8F4F3'), borderPadding=13, spaceBefore=13, spaceAfter=21))
story=[]
plain=[]
def p(text, style='BodyA'):
    story.append(Paragraph(text, styles[style])); plain.append(text)
def page(kicker, title):
    if story: story.append(PageBreak())
    p(kicker.upper(), 'SmallA'); p(title,'HeadA')
def table(headers, rows, widths):
    data=[[Paragraph(str(t), styles['SmallA']) for t in r] for r in [headers]+rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCECEB')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),.5,colors.HexColor('#D7E1E6'))]))
    story.append(t); story.append(Spacer(1,12))
    plain.extend([' | '.join(map(str,r)) for r in [headers]+rows])

p('PROGRESS REPORT FOR TAs AND MENTORS  /  18 OCTOBER 2026', 'SmallA')
story.append(Spacer(1,22))
p('Finding breathing<br/>in thermal images', 'TitleA')
p('What we have done, what we have learned,<br/>and what needs to happen next', 'SubA')
story.append(Spacer(1,22))
p('Our aim', 'SubA')
p('Our aim is to build a <b>breathing-rate detector with breathing-region segmentation from thermal video</b>. The system should identify facial areas associated with breathing, follow them as the person moves, and use their changing image intensity to estimate breaths per minute.')
p('In everyday terms, the question is: <b>can we find a reliable breathing signal by watching the face, without attaching a sensor to the person?</b>')
p('We have built and tested components for data processing, breathing-rate estimation, facial tracking and segmentation. Reliable performance of the complete system still needs validation.', 'CallA')
p('Two papers guide our approach: <b>Kwon et al.</b> informs breathing-region segmentation, while <b>JoulesEye</b> informs tracking the nasal region during movement. We are using these ideas to develop and evaluate our own pipeline on the available dataset.')
p('The latest experiment reached an important limit: interruptions in the recordings left no sufficiently complete segments for the main segmentation comparison. We can explain why the evaluation stopped, but we cannot yet say whether the method improves breathing accuracy.')
p('This report brings together the work completed so far, the results we can support, and the next steps for discussion with our TAs and mentors.', 'SmallA')

page('01 / The work completed', 'From an initial pipeline<br/>to a clearer research direction')
p('We reviewed the data and the earlier results', 'SubA')
p('The supplied dataset contains five people, with one recording per person. Four recordings currently have usable frame/reference pairings. One remains excluded because its image and reference-record counts disagree. Exact synchronization and the meaning of the reference markers still need confirmation.')
p('We identified problems that made the old result difficult to trust: an assumed recording speed, time being shortened after removing frozen frames, a broad facial crop, and inconsistent breath counting. We introduced timestamp-preserving processing, explicit exclusions and shared evaluation rules to make comparisons more meaningful.')
p('We explored tracking and learning-based approaches', 'SubA')
p('We evaluated a simple reference baseline, frequency-based estimation, a CNN pattern-learning model, and a CNN + GRU model that also learns changes over time. We tested on a person left out of training, then repeated this across the four usable people. We also added CSRT tracking, inspired by JoulesEye, and compared fixed and tracked regions.')
p('We implemented and corrected the segmentation branch', 'SubA')
p('The current method follows the face, measures recent thermal changes, highlights likely breathing regions, and extracts a breathing signal. We corrected the direction used to align images and replaced a single fixed segmentation mask with masks that update throughout the recording.')
p('Following Kwon\'s approach, we calculate recent thermal variation, normalize it into a probability-like map and group pixels into candidate breathing regions. The code retains the original timing and records where reliable processing is unavailable.')
p('We made the work easier to inspect and present', 'SubA')
p('The original notebook now includes the executed code, saved results, example masks, limitations and next steps. The latest software checks recorded <b>25 passes and one skipped hardware-dependent test</b>. These checks support implementation correctness, not real-world breathing accuracy.')

page('02 / Breathing-rate experiments', 'What did the models<br/>and tracking add?')
p('We first established breathing-rate estimates from a fixed facial region, then tested whether tracking that region helped. The following comparison uses the same 77 eligible segments from four people. Error is measured in breaths per minute; lower is better.')
table(['Method','Fixed region','Tracked region'],[
('Simple baseline: typical training-set breathing rate','2.501','2.501'),
('Frequency-based signal estimate','10.618','10.809'),
('CNN: learns patterns in the extracted signal','3.577','3.268'),
('CNN + GRU: also models changes over time','2.986','2.969')],[293,95,95])
p('What we learned', 'SubA')
p('Tracking reduced CNN error, but the main CNN + GRU result changed by only <b>0.017 breaths per minute</b>, about 0.6%. The simple baseline still had lower error than either learned model in this comparison. It uses a typical rate from training data, so it is a useful benchmark rather than a detector of an individual\'s breathing.')
p('These figures are from the planned seed-7 run. Across three training initializations, average CNN error improved from <b>3.682 to 3.247</b> with tracking. The CNN + GRU average changed only from <b>6.803 to 6.795</b>; one initialization produced about 14.87 error for both variants. This shows substantial training instability.')
p('Following a region is not the same as following the nose', 'SubA')
p('Visual checks found that the tracker could drift towards the neck or eyes while still reporting success. This is why we need to assess tracking position as well as the final breathing-rate error.')
p('The tracking adaptation uses thermal images and manually initialized regions. It does not include JoulesEye\'s RGB-assisted initialization or detector-based recovery. The overlapping segments and repeated training runs do not increase the number of independent people.', 'SmallA')

page('03 / Segmentation experiments', 'What happened when we tested<br/>the corrected method?')
p('We processed the same four usable recordings in four ways. Each approach was assessed using the same planned 20-second segments and quality rules. A segment was excluded if too much of its signal was missing.')
table(['Approach','Segments passing the rules'],[
('A fixed face region, without segmentation','73 of 84'),
('A tracked and aligned region, without segmentation','65 of 84'),
('Correct alignment plus breathing-region segmentation','0 of 84'),
('Deliberately wrong alignment plus segmentation, for comparison','0 of 84')],[325,158])
p('Why did the segmentation comparison stop?', 'SubA')
p('The method needs about four seconds of recent usable images to decide which pixels are changing. Frozen frames and tracking interruptions create gaps. The extra history needed after those gaps leaves too little continuous support for the 20-second evaluation. None of the segmentation segments passed the declared rules.')
p('Zero eligible segments does not mean zero accuracy. It means the main accuracy result is unavailable.', 'CallA')
p('What can we say about alignment alone?', 'SubA')
p('On 65 segments shared by the two approaches without segmentation, a peak-counting method reduced average rate error from <b>6.26 to 5.04 breaths per minute</b> after alignment. A frequency-based method worsened from <b>10.12 to 10.54</b>. Lower error is better. These mixed results do not establish a benefit from segmentation.')
p('The 84 candidate segments overlap and come from only four people. They are not 84 independent participants. Results from this experiment should not be ranked directly against earlier model experiments that used different usable segments.', 'SmallA')

page('04 / What the findings mean', 'What is working,<br/>and what remains unproven?')
p('We should not say that MRF segmentation has failed', 'SubA')
p('MRF is the paper\'s way of grouping nearby pixels into regions. The older pipeline had several problems at once, and the corrected experiment did not produce an eligible accuracy comparison. Neither result isolates the quality of segmentation well enough to dismiss the method.')
p('A highlighted region is not automatically a breathing region', 'SubA')
p('Some inspected masks include cheeks and face edges as well as the nose. Movement and other image changes can also make pixels vary. We need independently marked examples to check whether the method is selecting the right areas, rather than relying only on how convincing a mask looks.')
p('Our data are different from the paper\'s data', 'SubA')
p('Kwon\'s study used 15 participants, specified camera views and a synchronized respiratory sensor waveform. Our project currently has four usable recordings and event annotations. This prevents a like-for-like comparison with the paper\'s published signal agreement and breathing-cycle results.')
p('Our adaptations need their own validation', 'SubA')
p('Our dataset and implementation differ from the reference studies. Some segmentation settings are not fully specified in Kwon\'s paper, and our JoulesEye-inspired tracking uses thermal images alone. We have documented these choices; their value must be demonstrated on our own data.')
p('The fair conclusion', 'SubA')
p('We now have a working experimental pipeline and measured comparisons for several rate-estimation approaches. We have also implemented segmentation and identified a data-coverage obstacle. We have not yet validated a reliable detector that combines accurate region selection, tracking and breathing-rate estimation.')
p('The next step is to make the core experiment evaluable, then judge the method on its results.', 'CallA')

page('05 / The path forward', 'What needs to be done next')
table(['Priority','Action','What completion would look like'],[
('1. Verify the recordings','Check frozen periods, timestamps and breath annotations against the original acquisition files. Resolve the excluded recording only if its alignment can be verified.','A documented record of which images and reference events can be trusted.'),
('2. Obtain continuous segments','Recover uninterrupted source video if available; otherwise collect new still recordings with synchronized references.','At least four seconds of usable preparation followed by a complete 20-second evaluation segment.'),
('3. Check the selected regions','Independently mark nose positions and expected breathing regions in representative images.','Evidence of where tracking drifts and where segmentation selects the correct area.'),
('4. Repeat the planned comparison','Compare the same aligned regions with and without segmentation, using rules agreed before examining errors.','Paired accuracy results, usable-segment counts and all exclusions reported together.'),
('5. Validate the complete detector','Address model instability; integrate validated tracking and segmentation with rate estimation. Add people, sessions and synchronized breathing references.','Reliable estimates on new recordings, with failures detected and reported.')],[73,231,179])
p('Suggested closing statement for a presentation', 'SubA')
p('"Our goal is a thermal-video breathing-rate detector that also identifies the facial regions associated with breathing. We have prepared the data, tested signal-based and learned estimators, added tracking, and implemented segmentation. Our next priorities are reliable continuous data, verified region selection and a fair evaluation of the combined system."')
p('Discussion with our TAs and mentors', 'SubA')
p('We would value guidance on access to original recordings and synchronized references, how to annotate breathing regions, and what accuracy and failure-reporting targets should define a successful project.', 'SmallA')

page('06 / Research foundations', 'Two references,<br/>one project objective')
p('Kwon et al.: identifying where breathing appears', 'SubA')
p('This study guides our segmentation branch. Its central idea is to examine short-term changes in thermal images and group pixels into breathing-associated facial regions. We implemented the four-second variation map, its normalization and the MRF grouping step, alongside corrected image alignment and signal extraction.')
p('Kwon et al. (2023), <i>Breathing-Associated Facial Region Segmentation for Thermal Camera-Based Indirect Breathing Monitoring</i>. <link href="https://doi.org/10.1109/JTEHM.2023.3295775" color="#147D86">DOI: 10.1109/JTEHM.2023.3295775</link>.', 'SmallA')
p('JoulesEye: keeping track of the nasal region', 'SubA')
p('This study guides our work on respiration sensing during movement. We borrowed its use of CSRT tracking to follow the nasal region and tested whether that helped our breathing-rate estimates. Our adaptation is thermal-only and still needs better drift detection and recovery.')
p('Adhikary et al. (2023), <i>JoulesEye: Energy Expenditure Estimation and Respiration Sensing from Thermal Imagery While Exercising</i>. <link href="https://sustainability-lab.github.io/jouleseye/" color="#147D86">Research project and paper: sustainability-lab.github.io/jouleseye</link>.', 'SmallA')
p('How the references fit our work', 'SubA')
p('The references provide ideas for selecting and following useful facial regions. Our project brings these ideas into a pipeline that also includes data-quality checks, signal-based estimation, learned models and controlled comparisons. Results published in either paper are not reported as our own performance.')
p('Evidence and further reading', 'SubA')
p('Project evidence: docs/thermal_respiration_walkthrough.ipynb; docs/kwon_reproduction_protocol.md; docs/kwon_experiment_results.md; docs/tracking_experiment_results.md. Detailed latest results: artifacts/kwon_experiment/evaluation.json. No new experiments were run to prepare this report.', 'SmallA')

def footer(c, doc):
    c.setStrokeColor(colors.HexColor('#D7E1E6')); c.line(56,43,539,43)
    c.setFont('Helvetica',8); c.setFillColor(muted)
    c.drawString(56,29,'THERMAL BREATHING RESEARCH  |  Project status - 18 October 2026')
    c.drawRightString(539,29,str(doc.page))

path=OUT/'thermal_breathing_final_report.pdf'
doc=SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=56,leftMargin=56,topMargin=51,bottomMargin=60,
    title='Thermal Breathing Research - Project Status and Next Steps',author='Thermal Breathing Research Project')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'thermal_breathing_final_report.txt').write_text('\n\n'.join(plain),encoding='utf-8')
print(path)
