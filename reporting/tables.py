"""tables.tex generator (v4 §15.2).

Reads eval_summary.json and writes a single .tex file with table 1 + 2.
For each cell in the Ours row: keep ``\textbf{x}`` if Ours is the best
in the column (min for rel-L2, max for R²), otherwise just ``x``.
"""
from __future__ import annotations

import json
import os.path as osp
from typing import Any

LATEX_HEADER = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage[margin=0.5in]{geometry}
\begin{document}
"""

LATEX_FOOTER = r"""\end{document}
"""

TABLE1_TEMPLATE = r"""
\begin{table*}[htbp]
\centering
\caption{Relative L2 Errors (\%) and Coefficient of Determination ($R^2$) on the DrivAerML Dataset}
\label{tab:main_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | cc | ccc | cc}
\toprule
\multirow{2}{*}{\textsc{Model}} & \multicolumn{2}{c|}{\textsc{Surface Fields ($\downarrow$)}} & \multicolumn{3}{c|}{\textsc{Volume Fields ($\downarrow$)}} & \multicolumn{2}{c}{\textsc{Global Coefficients ($\uparrow$)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-8}
& Pressure ($p_s$) & Shear Stress ($\tau$) & Pressure ($p_v$) & Velocity ($u$) & Vorticity ($\omega$) & Drag ($C_d$) $R^2$ & Lift ($C_l$) $R^2$ \\
\midrule
Transolver          & 4.81 & 8.95 & 7.74 & 6.78 & 38.4 & - & - \\
Transolver++        & 5.26 & -    & -    & 7.16 & 40.3 & - & - \\
Transolver-3        & 3.71 & 5.85 & 5.72 & 4.14 & -    & 0.972 & 0.985 \\
AB-UPT              & 3.82 & 7.29 & 6.08 & 5.93 & 35.1 & 0.963 & 0.975 \\
\midrule
\textsc{Ours}       & {P_S} & {TAU} & {P_V} & {U} & {OMEGA} & {CD} & {CL} \\
\bottomrule
\multicolumn{8}{p{1.1\textwidth}}{\vspace{2pt}\footnotesize \textit{Note:} Performance comparison on the DrivAerML benchmark under the standard random data split. Relative L2 errors are presented in percentages (\%). Vector fields (velocity $u$ and wall shear stress $\tau$) are evaluated using their full 3D Frobenius norm. Missing values (-) indicate that the corresponding metric was not reported in the original paper.}
\end{tabular}%
}
\end{table*}
"""

TABLE2_TEMPLATE = r"""
\begin{table*}[htbp]
\centering
\caption{Component-wise Relative L2 Errors (\%) Comparison with DoMINO on DrivAerML}
\label{tab:domino_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | cccc | ccccc}
\toprule
\multirow{2}{*}{\textsc{Model}$^*$} & \multicolumn{4}{c|}{\textsc{Surface Fields ($\downarrow$)}} & \multicolumn{5}{c}{\textsc{Volume Fields ($\downarrow$)}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-10}
& $p_s$ & $\tau_x$ & $\tau_y$ & $\tau_z$ & $p_v$ & $u_x$ & $u_y$ & $u_z$ & $\nu_t$ \\
\midrule
DoMINO              & 15.05 & 21.24 & 30.20 & 33.59 & 21.93 & 23.97 & 50.25 & 45.67 & 21.75 \\
\midrule
\textsc{Ours}       & {P_S} & {TX} & {TY} & {TZ} & {P_V} & {U_X} & {U_Y} & {U_Z} & {NUT} \\
\bottomrule
\multicolumn{10}{p{1.2\textwidth}}{\vspace{2pt}\footnotesize $^*$ \textit{Note on DoMINO Comparison:} DoMINO employs a different evaluation protocol. First, it calculates the relative L2 error for vector fields by decomposing them into independent spatial components ($x, y, z$) rather than using the overall vector magnitude. Second, DoMINO evaluates turbulent viscosity ($\nu_t$). Lastly, DoMINO utilizes a specific out-of-distribution (OOD) data split based on drag force ranges, whereas other models utilize a uniform random split. We report our component-wise metrics here to provide a direct structural comparison.}
\end{tabular}%
}
\end{table*}
"""


def _format_cell(value: float, baselines: list[float | None],
                  higher_is_better: bool, fmt: str = '.2f') -> str:
    """Return either ``\textbf{x}`` (Ours wins) or plain ``x`` (Ours loses)."""
    rest = [v for v in baselines if v is not None]
    if not rest:
        return rf'\textbf{{{value:{fmt}}}}'
    if higher_is_better:
        wins = value >= max(rest)
    else:
        wins = value <= min(rest)
    return rf'\textbf{{{value:{fmt}}}}' if wins else f'{value:{fmt}}'


def render_tables(summary: dict[str, Any], cfg: dict[str, Any],
                  out_path: str) -> str:
    """Materialize tables.tex with Ours numbers + \textbf logic."""
    t1 = summary['table1']
    t2 = summary['table2_domino']
    base1 = cfg['reporting']['baselines_table1']
    base2 = cfg['reporting']['baselines_table2']

    def col1(key: str) -> list[float | None]:
        return [b.get(key) for b in base1.values()]

    def col2(key: str) -> list[float | None]:
        return [b.get(key) for b in base2.values()]

    table1 = TABLE1_TEMPLATE.format(
        P_S=_format_cell(t1['p_s_rel_l2_pct'], col1('p_s'), False),
        TAU=_format_cell(t1['tau_rel_l2_pct'], col1('tau'), False),
        P_V=_format_cell(t1['p_v_rel_l2_pct'], col1('p_v'), False),
        U=  _format_cell(t1['u_rel_l2_pct'], col1('u'), False),
        OMEGA=_format_cell(t1['omega_rel_l2_pct'], col1('omega'), False),
        CD= _format_cell(t1['cd_r2'], col1('cd_r2'), True, fmt='.3f'),
        CL= _format_cell(t1['cl_r2'], col1('cl_r2'), True, fmt='.3f'),
    )
    table2 = TABLE2_TEMPLATE.format(
        P_S=_format_cell(t2['p_s_rel_l2_pct'], col2('p_s'), False),
        TX= _format_cell(t2['tau_x_rel_l2_pct'], col2('tau_x'), False),
        TY= _format_cell(t2['tau_y_rel_l2_pct'], col2('tau_y'), False),
        TZ= _format_cell(t2['tau_z_rel_l2_pct'], col2('tau_z'), False),
        P_V=_format_cell(t2['p_v_rel_l2_pct'], col2('p_v'), False),
        U_X=_format_cell(t2['u_x_rel_l2_pct'], col2('u_x'), False),
        U_Y=_format_cell(t2['u_y_rel_l2_pct'], col2('u_y'), False),
        U_Z=_format_cell(t2['u_z_rel_l2_pct'], col2('u_z'), False),
        NUT=_format_cell(t2['nut_rel_l2_pct'], col2('nut'), False),
    )
    body = LATEX_HEADER + table1 + table2 + LATEX_FOOTER
    with open(out_path, 'w') as f:
        f.write(body)
    return out_path


def print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print both tables to terminal (tandemv1 _print_eval_summary style)."""
    t1 = summary['table1']
    t2 = summary['table2_domino']
    print('=' * 78)
    print('Table 1 — main comparison (relative L2 %, R²)')
    print('-' * 78)
    for k, v in t1.items():
        print(f'  {k:32s} {v:9.4f}')
    print('-' * 78)
    print('Table 2 — DoMINO component-wise comparison (relative L2 %)')
    print('-' * 78)
    for k, v in t2.items():
        print(f'  {k:32s} {v:9.4f}')
    print('=' * 78)
    if 'timing' in summary:
        print('Timing:')
        for k, v in summary['timing'].items():
            print(f'  {k:32s} {v:9.2f} s')
    if 'model' in summary:
        print('Model:')
        for k, v in summary['model'].items():
            print(f'  {k:32s} {v}')
    print('=' * 78)
