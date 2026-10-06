import type { ReactNode } from 'react'
import * as B from './basic'
import * as M from './ml'

/** Registry: curriculum `visual` key → interactive widget. */
export const WIDGETS: Record<string, { title: string; render: () => ReactNode }> = {
  ai_venn: { title: 'Where AI, ML and deep learning sit', render: () => <B.AiVenn /> },
  var_boxes: { title: 'Variables are labelled boxes', render: () => <B.LoopTracer boxes /> },
  loop_tracer: { title: 'Step through real Python', render: () => <B.LoopTracer /> },
  function_plot: { title: 'Play with a function', render: () => <B.FunctionPlot /> },
  vector_plot: { title: 'Vectors and the dot product', render: () => <B.VectorPlot /> },
  activation_plot: { title: 'Activation functions', render: () => <B.ActivationPlot /> },
  neuron_playground: { title: 'One artificial neuron', render: () => <B.NeuronPlayground /> },
  gd_descent: { title: 'Gradient descent, step by step', render: () => <B.GdDescent /> },
  nn_diagram: { title: 'Build a network', render: () => <B.NnDiagram /> },
  distribution_plot: { title: 'Distributions', render: () => <B.DistributionPlot /> },
  histogram: { title: 'Histograms of real data', render: () => <B.HistogramWidget /> },
  table_peek: { title: 'What a dataset looks like', render: () => <B.TablePeek /> },
  missing_map: { title: 'Where values are missing', render: () => <B.MissingMap /> },
  onehot: { title: 'One-hot encoding', render: () => <B.OneHot /> },
  scaling_demo: { title: 'Feature scaling', render: () => <B.ScalingDemo /> },
  imbalance_bar: { title: 'Class imbalance', render: () => <B.ImbalanceBar /> },
  scatter_line: { title: 'Fit a regression line yourself', render: () => <M.ScatterLine /> },
  decision_boundary: { title: 'Decision boundaries of real models', render: () => <M.DecisionBoundaryWidget /> },
  split_viz: { title: 'Train/test split', render: () => <M.SplitViz /> },
  cv_folds: { title: 'k-fold cross-validation', render: () => <M.CvFolds /> },
  overfit_curve: { title: 'Underfitting vs overfitting', render: () => <M.OverfitCurve /> },
  cm_threshold: { title: 'Interactive confusion matrix', render: () => <M.CmThreshold /> },
  kmeans_steps: { title: 'k-means, iteration by iteration', render: () => <M.KmeansSteps /> },
  conv_filter: { title: 'Convolution filters', render: () => <M.ConvFilter /> },
  tokenizer: { title: 'Train-and-use a BPE tokenizer', render: () => <M.TokenizerWidget /> },
  attention_heatmap: { title: 'Attention weights', render: () => <M.AttentionHeatmap /> },
  ngram_lm: { title: 'A tiny language model', render: () => <M.NgramLm /> },
  embedding_map: { title: 'Embeddings in 2D', render: () => <M.EmbeddingMap /> },
  chunker: { title: 'Chunking documents', render: () => <M.Chunker /> },
}

export function Widget({ name }: { name: string }) {
  const w = WIDGETS[name]
  if (!w) return <div className="muted">Visual “{name}” is not available.</div>
  return <div className="widget">{w.render()}</div>
}
