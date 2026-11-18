"""
PathSnap - Technical Report PDF Generator
Produces a publication-grade, fully compliant technical report PDF following PES University UE24CS352A guidelines.
Updated with official Evaluator-Testing Partition (evaluator_test) experimental results.
"""

import os
import sys
import shutil
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "report", "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PDF_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "report", "PathSnap_Technical_Report.pdf")
DOWNLOADS_OUTPUT_PATH = r"C:\Users\Anirudha\Downloads\PathSnap_Technical_Report.pdf"

# Custom NumberedCanvas for professional "Page X of Y" and running headers
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))

        # Skip running header and footer on Page 1 (Cover Page)
        if self._pageNumber > 1:
            # Running Header
            self.drawString(54, 11 * inch - 36, "PES University | UE24CS352A: Machine Learning — PathSnap Technical Report")
            self.setStrokeColor(colors.HexColor("#D1D5DB"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

            # Running Footer
            page_str = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(8.5 * inch - 54, 32, page_str)
            self.drawString(54, 32, "CONFIDENTIAL & PROPRIETARY — ACADEMIC EVALUATION")
            self.line(54, 44, 8.5 * inch - 54, 44)

        self.restoreState()


def build_pdf_report():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Define clean, highly legible styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=21,
        leading=25,
        textColor=colors.HexColor('#1E3A8A'),
        alignment=1, # Center
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#4B5563'),
        alignment=1,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=13,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1F2937'),
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1F2937'),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2.5
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#4B5563'),
        alignment=1,
        spaceBefore=3,
        spaceAfter=8
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1E3A8A')
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1F2937')
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # =========================================================
    # COVER / HEADER BANNER
    # =========================================================
    story.append(Spacer(1, 5))
    story.append(Paragraph("PES UNIVERSITY — RR CAMPUS", subtitle_style))
    story.append(Paragraph("Department of Computer Science and Engineering | UE24CS352A: Machine Learning", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceBefore=1, spaceAfter=10))
    
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("Final Technical Report & Evaluator-Test Partition Experimental Evaluation", subtitle_style))
    
    # Metadata Box
    meta_data = [
        [
            Paragraph("<b>Target Hackathon Threshold:</b> > 90.0% Accuracy", table_cell),
            Paragraph("<b>Evaluator-Test Top-1 Accuracy:</b> <b>94.58%</b> (Achieved)", table_cell)
        ],
        [
            Paragraph("<b>Student-Test Top-1 Accuracy:</b> <b>93.65%</b>", table_cell),
            Paragraph("<b>Evaluator-Test Route Continuity:</b> <b>98.42%</b>", table_cell)
        ],
        [
            Paragraph("<b>Primary Classifier Model:</b> Bayesian-Tuned Random Forest", table_cell),
            Paragraph("<b>Sequential Decoder:</b> Viterbi Hidden Markov Model (HMM)", table_cell)
        ],
        [
            Paragraph("<b>Evaluator Dataset Observations:</b> 19,674 GPS Points", table_cell),
            Paragraph("<b>Deployment Mode:</b> Distributed Two-Laptop LAN Architecture", table_cell)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # Executive Summary Highlight Box
    exec_summary_text = (
        "<b>Executive Summary:</b> PathSnap is an end-to-end distributed machine learning system that reconstructs "
        "accurate, realistic vehicle routes from noisy urban GPS trajectories. The pipeline pairs spatial-grid candidate "
        "generation (99.97% recall), a 13-dimensional kinematic and geometric feature set, a Bayesian-tuned Random Forest classifier, "
        "and a topology-constrained Viterbi Hidden Markov Model (HMM). When evaluated on the official reserved "
        "<b>evaluator_test partition (19,674 GPS observations across 10 trips)</b>, PathSnap achieved an overall "
        "<b>Pointwise Top-1 Accuracy of 94.58%</b> and a <b>Route Continuity of 98.42%</b>, exceeding the competition benchmark "
        "while delivering sub-second inference over a two-laptop distributed architecture."
    )
    t_exec = Table([[Paragraph(exec_summary_text, callout_style)]], colWidths=[504])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
        ('BOX', (0,0), (-1,-1), 1.2, colors.HexColor('#3B82F6')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 9),
        ('RIGHTPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(t_exec)
    story.append(Spacer(1, 8))

    # =========================================================
    # SECTION 2: DATASET UNDERSTANDING & EDA
    # =========================================================
    story.append(Paragraph("2. Dataset Understanding and Exploratory Data Analysis", h1_style))
    story.append(Paragraph(
        "<b>2.1 Preprocessing and Candidate Generation:</b> Trajectory coordinates reflect urban multipath effects, GPS drift, "
        "and sensor jitter. Road networks are structured as directed OpenStreetMap graphs with node coordinates and directed arcs (u &rarr; v). "
        "We implemented a fast Spatial Grid Index (cell size 0.0008&deg; &approx; 89m) querying candidate arcs within a 50m search radius. "
        "Candidate generation achieved a <b>99.97% recall</b>. Ground-truth route mapping uses an adaptive forward window with global fallback, "
        "preventing window desynchronization during vehicle detours or loops.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.2 Data Partitions & Missing-Value Analysis:</b> Three distinct partitions were processed: "
        "(1) <i>train</i> (80 trips, 194,547 GPS points) used for feature extraction, 5-fold cross-validation, and Bayesian tuning; "
        "(2) <i>student_test</i> (10 trips, 19,160 GPS points) used for development benchmarking; and "
        "(3) <i>evaluator_test</i> (10 reserved trips, 19,674 GPS points) used for final evaluation. "
        "Coordinate completeness is 100% across all partitions with zero missing values.",
        body_style
    ))

    # EDA Figure
    eda_fig = os.path.join(FIGURES_DIR, "fig1_eda_distributions.png")
    if os.path.exists(eda_fig):
        story.append(Image(eda_fig, width=6.8*inch, height=2.3*inch))
        story.append(Paragraph("<b>Figure 1:</b> Exploratory Data Analysis of Trajectory Kinematics, Arc Densities, and Spatial Noise Distributions.", caption_style))

    story.append(Paragraph(
        "<b>2.3 Extracted 13-Dimensional Feature Hierarchy:</b> Thirteen geometric, kinematic, and rank features were engineered: "
        "<i>perp_dist</i> (orthogonal projection distance), <i>dist_start</i> &amp; <i>dist_end</i> (distances to arc endpoints), "
        "<i>t_ratio</i> (clamped projection parameter [0, 1]), <i>arc_len</i> (road segment length), <i>gps_speed</i> (instantaneous velocity), "
        "<i>time_delta</i> (sampling interval), <i>heading_diff</i> &amp; <i>cos_heading_diff</i> (angular alignment with road bearing), "
        "<i>spatial_prior</i> (Gaussian spatial likelihood), <i>dist_rank</i> (candidate distance rank), <i>dist_ratio_to_best</i> (ratio to nearest candidate), "
        "and <i>dist_diff_to_best</i> (orthogonal delta to closest candidate).",
        body_style
    ))

    # =========================================================
    # SECTION 3: MODEL EXPERIMENTATION & COMPARISON
    # =========================================================
    story.append(Paragraph("3. Model Experimentation and Cross-Family Comparison", h1_style))
    story.append(Paragraph(
        "We implemented, trained, and benchmarked <b>five distinct machine learning model families</b> using 5-Fold GroupKFold "
        "cross-validation grouped by trip ID on the 80 training trips, followed by standardized testing on student_test.",
        body_style
    ))

    # Table 3.1
    story.append(Paragraph("<b>Table 3.1: Consolidated Cross-Family Benchmark Comparison Table</b>", h2_style))
    model_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_models = pd.read_csv(model_csv)

    table_data = [[
        Paragraph("<b>Model</b>", table_header),
        Paragraph("<b>Algorithm Family</b>", table_header),
        Paragraph("<b>Key Configuration</b>", table_header),
        Paragraph("<b>CV Macro F1</b>", table_header),
        Paragraph("<b>Student Top-1 Acc</b>", table_header),
        Paragraph("<b>Route Continuity</b>", table_header),
        Paragraph("<b>Log Loss</b>", table_header),
        Paragraph("<b>Latency (ms/1k)</b>", table_header)
    ]]

    for _, row in df_models.iterrows():
        table_data.append([
            Paragraph(f"<b>{row['Model Name']}</b>", table_cell),
            Paragraph(str(row['Algorithm Family']), table_cell),
            Paragraph(str(row['Key Configuration']), table_cell),
            Paragraph(f"{row['CV Macro F1']:.4f}", table_cell),
            Paragraph(f"{row['Student-Test Top-1 Acc']*100:.2f}%", table_cell),
            Paragraph(f"{row['Student-Test Continuity']*100:.2f}%", table_cell),
            Paragraph(f"{row['Log Loss']:.3f}", table_cell),
            Paragraph(f"{row['Inference Time (ms/1k)']:.2f}", table_cell)
        ])

    t_models = Table(table_data, colWidths=[70, 75, 110, 48, 56, 52, 42, 51])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 4))

    # Benchmark Figures
    bench_fig = os.path.join(FIGURES_DIR, "fig2_model_benchmark_bars.png")
    if os.path.exists(bench_fig):
        story.append(Image(bench_fig, width=6.8*inch, height=2.2*inch))
        story.append(Paragraph("<b>Figure 2:</b> Cross-Validation Macro F1, Student-Test Macro F1, and Observation Top-1 Accuracy across Models.", caption_style))

    story.append(Paragraph("<b>3.2 Compulsory Questions for EVERY Individual Model:</b>", h2_style))

    # Model 1
    story.append(Paragraph("<b>Model 1: Logistic Regression (Linear & Probabilistic)</b>", h3_style))
    story.append(Paragraph(
        "&bull; <i>Suitability & Strengths/Limitations:</i> Convex baseline with calibrated posterior probabilities. Extremely fast (0.26 ms/1k obs), but linear hyperplanes struggle with nonlinear heading/distance spatial trade-offs.<br/>"
        "&bull; <i>Training & Results:</i> L2 penalty, C=1.0, max_iter=1000. CV Macro F1: 0.8816, Student Top-1: 70.54%, Log Loss: 0.134. Slight underfitting due to linear capacity constraints.",
        bullet_style
    ))

    # Model 2
    story.append(Paragraph("<b>Model 2: Decision Tree (Tree-based)</b>", h3_style))
    story.append(Paragraph(
        "&bull; <i>Suitability & Strengths/Limitations:</i> Orthogonal splits isolate spatial boundaries naturally (e.g. perp_dist &lt; 15m AND heading_diff &lt; 30&deg;). Fast, but stepped probabilities increase variance.<br/>"
        "&bull; <i>Training & Results:</i> Gini impurity, max_depth=12, min_samples_leaf=10. CV Macro F1: 0.8811, Student Top-1: 71.07%, Log Loss: 0.134. Controlled leaf sizes prevented severe overfitting.",
        bullet_style
    ))

    # Model 3
    story.append(Paragraph("<b>Model 3: Random Forest (Ensemble / Bagging) — [SELECTED PRIMARY MODEL]</b>", h3_style))
    story.append(Paragraph(
        "&bull; <i>Suitability & Strengths/Limitations:</i> Bootstrap aggregation of decorrelated trees reduces variance drastically while learning complex multi-modal geometry. Superior generalization, robust to GPS outliers, smooth probabilistic outputs.<br/>"
        "&bull; <i>Training & Results:</i> 100 trees, max_depth=15, min_samples_leaf=5. <b>Highest Student Top-1 Accuracy (71.26%) and Lowest Log Loss (0.121)</b> with 95.00% continuity. Generalization gap between CV (0.8872) and test F1 (0.8633) confirms optimal regularization.",
        bullet_style
    ))

    # Model 4
    story.append(Paragraph("<b>Model 4: HistGradientBoosting (Boosting)</b>", h3_style))
    story.append(Paragraph(
        "&bull; <i>Suitability & Strengths/Limitations:</i> Histogram binning enables fast gradient boosting. High CV Macro F1 (0.8961), but sharp boundary probabilities slightly degraded HMM emission blending.<br/>"
        "&bull; <i>Training & Results:</i> 100 iterations, lr=0.1, max_depth=8. Student Top-1: 71.13%, Test F1: 0.8615, Log Loss: 0.123.",
        bullet_style
    ))

    # Model 5
    story.append(Paragraph("<b>Model 5: K-Nearest Neighbors (Instance-based)</b>", h3_style))
    story.append(Paragraph(
        "&bull; <i>Suitability & Strengths/Limitations:</i> Non-parametric similarity. Proved unsuitable due to high inference latency (349.3 ms/1k) and poor trajectory continuity (49.95%) caused by high-dimensional distance distortion.<br/>"
        "&bull; <i>Training & Results:</i> k=7, distance-weighted, Euclidean metric. Top-1 Accuracy: 16.77%, Log Loss: 0.699.",
        bullet_style
    ))

    lat_fig = os.path.join(FIGURES_DIR, "fig3_latency_and_logloss.png")
    if os.path.exists(lat_fig):
        story.append(Image(lat_fig, width=6.8*inch, height=2.0*inch))
        story.append(Paragraph("<b>Figure 3:</b> Inference Latency (ms/1,000 obs, log scale) and Log Loss Probability Calibration across Models.", caption_style))

    # =========================================================
    # SECTION 4: BAYESIAN OPTIMIZATION
    # =========================================================
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation", h1_style))
    story.append(Paragraph(
        "<b>Random Forest</b> was selected for Bayesian hyperparameter optimization using the Optuna framework with Tree-structured "
        "Parzen Estimators (TPE) across the 80 training trips.",
        body_style
    ))
    story.append(Paragraph(
        "<b>4.1 Search Space & Objective:</b> Evaluated: <i>n_estimators</i> &isin; [100, 300], <i>max_depth</i> &isin; [8, 25], "
        "<i>min_samples_split</i> &isin; [2, 10], <i>min_samples_leaf</i> &isin; [1, 8], and <i>max_features</i> &isin; [None, 'sqrt', 'log2']. "
        "Optimized objective: <b>Composite Score = 0.60 &times; Top1_Accuracy + 0.40 &times; Route_Continuity</b>.",
        body_style
    ))

    opt_fig = os.path.join(FIGURES_DIR, "fig5_bayesian_optimization.png")
    if os.path.exists(opt_fig):
        story.append(Image(opt_fig, width=6.8*inch, height=2.1*inch))
        story.append(Paragraph("<b>Figure 4:</b> Bayesian Optimization Trial Progression (TPE Sampler) and Hyperparameter Landscape Search.", caption_style))

    story.append(Paragraph(
        "<b>4.2 Winning Configuration:</b> Trial 2 yielded the winning parameters: "
        "<b>n_estimators=100, max_depth=12, min_samples_split=4, min_samples_leaf=5, max_features=None</b> (score = 0.8017). "
        "Refitting on all 80 training trips reduced tree depth from 15 to 12, improving route continuity to <b>95.09%</b> at 3.5 ms/1k latency.",
        body_style
    ))

    feat_fig = os.path.join(FIGURES_DIR, "fig4_feature_importance.png")
    if os.path.exists(feat_fig):
        story.append(Image(feat_fig, width=6.8*inch, height=2.1*inch))
        story.append(Paragraph("<b>Figure 5:</b> Random Forest Feature Importance Breakdown. Distance ratio to best and angular heading difference drive 87.2% of decision splits.", caption_style))

    # =========================================================
    # SECTION 5: HMM & VITERBI ROUTE RECONSTRUCTION
    # =========================================================
    story.append(Paragraph("5. HMM and Viterbi Route Reconstruction", h1_style))
    story.append(Paragraph(
        "Pointwise classifier predictions lack temporal continuity: consecutive observations can jump between parallel roads. "
        "We implemented a Hidden Markov Model with dynamic programming Viterbi decoding to enforce graph connectivity.",
        body_style
    ))
    story.append(Paragraph(
        "<b>5.1 Mathematical Formulation:</b><br/>"
        "&bull; <i>Emission Probability:</i> Blends Gaussian spatial distance likelihood and Random Forest posterior probabilities: "
        "<b>P(o_t | s_t = a_i) &prop; exp(-d(o_t, a_i)&sup2; / (2&sigma;_z&sup2;)) &times; [P_rf(y=1 | x_t,i)]</b> with &sigma;_z = 8.0m.<br/>"
        "&bull; <i>Transition Probability:</i> Enforces OpenStreetMap network topology. If <i>a_i == a_j</i>, P = 0.88 (persistence). "
        "If <i>v(a_i) == u(a_j)</i>, P = 1.0 (connected junction). Non-adjacent transitions decay exponentially with shortest-path distance: exp(-|d_net - d_gps| / &beta;) with &beta; = 12.0m.<br/>"
        "&bull; <i>Temporal Gap Handling:</i> If sampling interval &Delta;t > 5.0s (e.g. tunnel GPS blackout), transition priors reset to uniform.",
        body_style
    ))

    # Evaluator Test Results Table
    story.append(Paragraph("<b>Table 5.1: Performance on Official Evaluator-Testing Partition (evaluator_test)</b>", h2_style))
    eval_csv = os.path.join(RESULTS_DIR, "hmm_vs_classifier_evaluator_test.csv")
    df_eval = pd.read_csv(eval_csv)

    eval_table_data = [[
        Paragraph("<b>Trip ID</b>", table_header),
        Paragraph("<b>Points</b>", table_header),
        Paragraph("<b>Classifier Acc</b>", table_header),
        Paragraph("<b>HMM Acc</b>", table_header),
        Paragraph("<b>Acc Diff</b>", table_header),
        Paragraph("<b>Clf Continuity</b>", table_header),
        Paragraph("<b>HMM Continuity</b>", table_header),
        Paragraph("<b>Time (ms)</b>", table_header)
    ]]

    for _, row in df_eval.iterrows():
        t_id_str = f"000000{int(row['trip_id']):02d}" if isinstance(row['trip_id'], (int, float)) else str(row['trip_id'])
        diff_str = f"+{row['acc_improvement']*100:.2f}%" if row['acc_improvement'] >= 0 else f"{row['acc_improvement']*100:.2f}%"
        eval_table_data.append([
            Paragraph(t_id_str, table_cell),
            Paragraph(f"{int(row['observations']):,}", table_cell),
            Paragraph(f"{row['clf_only_accuracy']*100:.2f}%", table_cell),
            Paragraph(f"<b>{row['hmm_accuracy']*100:.2f}%</b>", table_cell),
            Paragraph(diff_str, table_cell),
            Paragraph(f"{row['clf_continuity']*100:.2f}%", table_cell),
            Paragraph(f"<b>{row['hmm_continuity']*100:.2f}%</b>", table_cell),
            Paragraph(f"{row['decode_time_ms']:.0f}", table_cell)
        ])

    # Overall Summary Row
    overall_obs_eval = df_eval['observations'].sum()
    weighted_clf_acc_eval = (df_eval['clf_only_accuracy'] * df_eval['observations']).sum() / overall_obs_eval
    weighted_hmm_acc_eval = (df_eval['hmm_accuracy'] * df_eval['observations']).sum() / overall_obs_eval
    mean_clf_cont_eval = df_eval['clf_continuity'].mean()
    mean_hmm_cont_eval = df_eval['hmm_continuity'].mean()
    total_time_eval = df_eval['decode_time_ms'].sum()

    eval_table_data.append([
        Paragraph("<b>OVERALL</b>", table_header),
        Paragraph(f"<b>{overall_obs_eval:,}</b>", table_header),
        Paragraph(f"<b>{weighted_clf_acc_eval*100:.2f}%</b>", table_header),
        Paragraph(f"<b>{weighted_hmm_acc_eval*100:.2f}%</b>", table_header),
        Paragraph(f"<b>+{(weighted_hmm_acc_eval - weighted_clf_acc_eval)*100:.2f}%</b>", table_header),
        Paragraph(f"<b>{mean_clf_cont_eval*100:.2f}%</b>", table_header),
        Paragraph(f"<b>{mean_hmm_cont_eval*100:.2f}%</b>", table_header),
        Paragraph(f"<b>{total_time_eval/1000:.1f} s</b>", table_header)
    ])

    t_eval = Table(eval_table_data, colWidths=[65, 50, 68, 65, 52, 68, 76, 60])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#1E40AF')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 4))

    hmm_comp_fig = os.path.join(FIGURES_DIR, "fig6_hmm_vs_classifier.png")
    if os.path.exists(hmm_comp_fig):
        story.append(Image(hmm_comp_fig, width=6.8*inch, height=2.4*inch))
        story.append(Paragraph("<b>Figure 6:</b> Per-Trip Accuracy and Continuity across the 10 Official Evaluator-Testing Trips (evaluator_test).", caption_style))

    # Partition Consistency Box
    partition_comp_text = (
        "<b>Cross-Partition Generalization Consistency:</b><br/>"
        "&bull; <b>Development Test Set (student_test, 10 Trips):</b> 93.65% Top-1 Accuracy | 99.22% Route Continuity<br/>"
        "&bull; <b>Official Evaluator Set (evaluator_test, 10 Trips):</b> <b>94.58% Top-1 Accuracy</b> | <b>98.42% Route Continuity</b><br/>"
        "This consistent high performance across both unseen test partitions confirms that PathSnap has not overfit and "
        "generalizes robustly to novel urban driving trajectories."
    )
    t_part = Table([[Paragraph(partition_comp_text, callout_style)]], colWidths=[504])
    t_part.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0FDF4')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#16A34A')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_part)
    story.append(Spacer(1, 6))

    # =========================================================
    # SECTION 6: GEOGRAPHICAL VISUALIZATION USING FOLIUM
    # =========================================================
    story.append(Paragraph("6. Geographical Visualisation Using Folium", h1_style))
    story.append(Paragraph(
        "Interactive Folium maps render four distinct togglable layers: "
        "(1) <b>Local Road Network Geometry</b> in light grey (#999999); "
        "(2) <b>Ground Truth Reference Route</b> in dashed green (#00BB00); "
        "(3) <b>Observed GPS Trajectory</b> with orange markers and polyline (#FF8800); and "
        "(4) <b>Reconstructed Route</b> in solid blue (#0044FF) reflecting the output of the Viterbi HMM.",
        body_style
    ))

    map_fig = os.path.join(FIGURES_DIR, "fig7_folium_map_rendering.png")
    if os.path.exists(map_fig):
        story.append(Image(map_fig, width=6.8*inch, height=2.8*inch))
        story.append(Paragraph("<b>Figure 7:</b> PathSnap Map-Matching Reconstruction on Evaluator Trip 00000015 (6,854 GPS Observations, 93.13% Accuracy, 99.97% Continuity).", caption_style))

    # =========================================================
    # SECTION 7: API, UI AND DEPLOYMENT
    # =========================================================
    story.append(Paragraph("7. API, UI and Two-Laptop Distributed Deployment", h1_style))
    story.append(Paragraph(
        "The system is deployed across two laptops communicating over local Wi-Fi / LAN:",
        body_style
    ))
    story.append(Paragraph(
        "&bull; <b>Laptop 1 (Model & Inference Backend):</b> Hosts FastAPI server on port 8000. "
        "Loads StandardScaler, Random Forest classifier, and Viterbi decoder. Exposes <code>GET /api/trajectories</code> "
        "and <code>POST /api/map-match</code>. Evaluates accuracy server-side so reference routes remain confidential.<br/>"
        "&bull; <b>Laptop 2 (Evaluator Dashboard UI):</b> Runs Streamlit interface on port 8501. "
        "Allows entering Laptop 1 IP address, displays the 10 evaluator trips in a dropdown, and renders live accuracy, "
        "route continuity, and interactive Folium maps.",
        bullet_style
    ))

    arch_fig = os.path.join(FIGURES_DIR, "fig8_system_architecture.png")
    if os.path.exists(arch_fig):
        story.append(Image(arch_fig, width=6.8*inch, height=2.2*inch))
        story.append(Paragraph("<b>Figure 8:</b> Two-Laptop Distributed Map-Matching Architecture and LAN Communication Protocol.", caption_style))

    # =========================================================
    # SECTION 8: ACADEMIC INTEGRITY DECLARATION
    # =========================================================
    story.append(Paragraph("8. Academic Integrity and Team Verification", h1_style))
    story.append(Paragraph(
        "<b>8.1 Academic Honesty Declaration:</b><br/>"
        "We hereby certify that the code, experiments, analysis, and report submitted for PathSnap are the original and independent "
        "work of this team. No model weights, hyperparameter configurations, or source code were shared with or copied from any other team. "
        "All metrics reported in this technical document (including the 94.58% Evaluator Top-1 Accuracy and 98.42% Route Continuity) "
        "are strictly verifiable by executing the reproducible pipeline scripts in the submitted GitHub repository.",
        body_style
    ))

    # Sign-off box
    sign_data = [
        [
            Paragraph("<b>Evaluator Verification Checklist:</b>", table_cell),
            Paragraph("<b>Status</b>", table_header)
        ],
        [
            Paragraph("1. Minimum 5 Models Evaluated Across Families", table_cell),
            Paragraph("COMPLETED (5 Families)", table_cell)
        ],
        [
            Paragraph("2. Bayesian Hyperparameter Optimization Conducted", table_cell),
            Paragraph("COMPLETED (Optuna TPE)", table_cell)
        ],
        [
            Paragraph("3. Observation Top-1 Accuracy Target (>90%)", table_cell),
            Paragraph("<b>ACHIEVED: 94.58% (Evaluator Test)</b>", table_cell)
        ],
        [
            Paragraph("4. Route Continuity Metric", table_cell),
            Paragraph("<b>ACHIEVED: 98.42% (Evaluator Test)</b>", table_cell)
        ],
        [
            Paragraph("5. Interactive Folium Visualization Implemented", table_cell),
            Paragraph("COMPLETED", table_cell)
        ],
        [
            Paragraph("6. Distributed Two-Laptop LAN API & UI Tested", table_cell),
            Paragraph("VERIFIED & ONLINE", table_cell)
        ]
    ]
    t_sign = Table(sign_data, colWidths=[360, 144])
    t_sign.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_sign)

    # Build the document
    print(f"Building PDF report at {PDF_OUTPUT_PATH}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF build completed successfully!")

    # Copy to Downloads folder
    shutil.copyfile(PDF_OUTPUT_PATH, DOWNLOADS_OUTPUT_PATH)
    print(f"Copied updated PDF to Downloads at {DOWNLOADS_OUTPUT_PATH}!")

if __name__ == "__main__":
    build_pdf_report()
