\subsubsection*{Linear Model for Joint Effect Estimation}

To quantify the joint effects of wild-type (WT) variants and clinical variants on variant effect predictor (VEP) scores, we used regularized linear regression via \texttt{sklearn.linear\_model.Ridge} (L2 regularization). For each gene, we constructed a binary matrix $\mathbf{X}$ (\#haplotypes $\times$ \#WT variants), representing the presence or absence of each WT variant in each haplotype, and a continuous matrix $\mathbf{Y}$ (\#haplotypes $\times$ \#sites) containing VEP scores for each haplotype-site pair. We then fit a multi-target Ridge regression model:

\[
\mathbf{Y} = \mathbf{X}\mathbf{\beta} + \varepsilon
\]

Here, $\mathbf{\beta}$ (a WT variants $\times$ sites coefficient matrix, as returned by \texttt{Ridge.coef\_}) is fit by minimizing squared error plus a regularization penalty established by the strength parameter $\alpha$. The absolute value of each coefficient, $|\beta_{ij}|$, measures the magnitude of the joint effect between WT variant $i$ and site $j$, and the sign of $\beta_{ij}$ encodes the direction of the effect (negative values indicating more pathogenic, positive indicating more benign).

For each WT variant-site pair, we computed:
\begin{enumerate}
    \item Absolute joint effect magnitude $|\beta_{ij}|$
    \item Signed joint effect $\beta_{ij}$
    \item The number of haplotypes containing the WT variant (computed by summing the corresponding binary indicator column)
    \item Weighted joint effect: $\sqrt{|\beta_{ij}| \times n_\mathrm{haplotypes}}$ (to balance effect size and sample count)
\end{enumerate}
Model quality was evaluated using standard metrics from \texttt{sklearn.metrics}: $R^2$, explained variance, mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE).

\subsubsection*{Epistasis Testing}

To identify truly epistatic (non-additive) interactions (as opposed to additive effects), we implemented a statistical testing procedure that directly compares observed joint effects to the sum of their separate effects.

For each WT variant $i$ and clinical variant $j$ pair, we first computed:

\begin{enumerate}
    \item \textbf{Individual WT variant effect:} The mean VEP difference between haplotypes where WT variant $i$ is present (WT=1) vs. absent (WT=0), averaging across all clinical variants:
    \[
    \text{Effect}_{\mathrm{WT},\,i} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left( \overline{\mathrm{VEP}}(\mathrm{WT}_i{=}1,\, s) - \overline{\mathrm{VEP}}(\mathrm{WT}_i{=}0,\, s) \right)
    \]
    where $\mathcal{S}$ is the set of all clinical variant sites.

    \item \textbf{Individual clinical variant effect:} The average VEP for clinical variant $j$ relative to the overall VEP baseline (mean across all haplotypes and all clinical variants):
    \[
    \text{Effect}_{\mathrm{Clinical},\,j} = \overline{\mathrm{VEP}}(\mathrm{Clinical}_j) - \overline{\mathrm{VEP}}_\mathrm{baseline}
    \]
\end{enumerate}

The \textbf{expected additive effect} is then:
\[
\text{Effect}_\mathrm{additive} = \text{Effect}_{\mathrm{WT},\,i} + \text{Effect}_{\mathrm{Clinical},\,j}
\]

The \textbf{observed joint effect} is given by the corresponding $\beta_{ij}$ coefficient from the main \texttt{Ridge} model, which reflects the effect when both WT variant $i$ and clinical variant $j$ are present.

To statistically assess whether this joint effect is epistatic, we compared two nested models, both fit via \texttt{sklearn.linear\_model.Ridge}:

\begin{itemize}
\item \textbf{Additive model:}
\[
\mathrm{VEP} = \beta_0 + \beta_1 \cdot \mathrm{WT}
\]
where the WT variant's effect is constant (clinical variant is absorbed in the intercept).

\item \textbf{Interaction model:}
\[
\mathrm{VEP} = \beta_0 + \beta_1 \cdot \mathrm{WT} + \beta_2 \cdot \left(\mathrm{WT} \times \Delta_\mathrm{deviation}\right)
\]
where $\Delta_\mathrm{deviation} = \text{Effect}_\mathrm{additive} - \text{Effect}_{\mathrm{WT},\,i}$, capturing deviation from pure additivity, with $\beta_2$ representing the epistatic interaction strength.
\end{itemize}

Model fits for both cases were obtained using \texttt{Ridge} (with the same regularization setup), and the null hypothesis ($\beta_2 = 0$, i.e., no interaction/epistasis) was tested via an F-test comparing the residual sum of squares ($\mathrm{RSS}$):

\[
F = \frac{(RSS_\mathrm{additive} - RSS_\mathrm{interaction}) / (df_\mathrm{additive} - df_\mathrm{interaction})}{RSS_\mathrm{interaction} / df_\mathrm{interaction}}
\]

where $df_\mathrm{additive} = n - 2$ and $df_\mathrm{interaction} = n - 3$ ($n =$ number of haplotypes). P-values were calculated using \texttt{scipy.stats.f.cdf} (F-distribution, $1$ and $df_\mathrm{interaction}$ degrees of freedom). Since \texttt{Ridge} is a regularized model, degrees of freedom are approximate, but the F-test remains valid for comparing nested models.

A pair was classified as epistatic if the interaction model fit led to a statistically significant improvement ($p < 0.05$ by default, controlled via the \texttt{epistasis\_pvalue\_threshold} parameter) over the additive-only model. For each test, we recorded:

\begin{enumerate}
    \item Epistasis $p$-value and F-statistic
    \item $R^2$ and MSE for both the additive and interaction models
    \item $R^2$ improvement ($\Delta R^2 = R^2_\mathrm{interaction} - R^2_\mathrm{additive}$)
    \item Interaction coefficient $\beta_2$ (strength of epistasis)
    \item The joint effect $\beta_{ij}$, expected additive effect, and their deviation
    \item Individual WT variant and clinical variant effects
\end{enumerate}

To ensure robust inference, we restricted epistasis tests to variant pairs meeting the following criteria:

\begin{enumerate}
    \item At least 10 haplotypes with complete, non-missing data for both the WT variant and the clinical variant site
    \item Sufficient variation in WT variant status (std $> 10^{-10}$)
    \item Both WT=0 and WT=1 represented in the dataset
\end{enumerate}
These filters prevent false positives from insufficient or uninformative data.

\subsubsection*{Implementation}

All analyses were conducted in Python 3.x. Linear regression models (including both main interaction discovery and epistasis testing) were fit using \texttt{sklearn.linear\_model.Ridge}. Statistical hypothesis tests used \texttt{scipy.stats.f} functions. Data preprocessing, aggregation, and manipulation were performed with \texttt{pandas}. All described methods are implemented in the \texttt{wtvariants\_to\_vep\_linear\_model} function, which provides options for Ridge regression setup, regularization strength, and enabling epistasis testing. When \texttt{test\_epistasis=True}, this function returns an extended interaction dataframe containing epistasis annotations and metrics, as well as summary counts and mean effect sizes for additive and epistatic interactions.
