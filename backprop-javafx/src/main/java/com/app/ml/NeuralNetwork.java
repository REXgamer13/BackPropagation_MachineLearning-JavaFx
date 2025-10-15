package com.app.ml;

public class NeuralNetwork {
  private final int nI, nJ, nK;
  private final double alpha;

  // Pesos: wIJ[j][i], wJK[k][j]
  private final double[][] wIJ;
  private final double[][] wJK;

  public NeuralNetwork(int nI, int nJ, int nK, double alpha) {
    this.nI = nI; this.nJ = nJ; this.nK = nK; this.alpha = alpha;
    this.wIJ = new double[nJ][nI];
    this.wJK = new double[nK][nJ];
  }

  public void setWeightsIJ(double[][] w) {
    if (w.length != nJ || w[0].length != nI)
      throw new IllegalArgumentException("Dimensiones de wIJ no coinciden");
    for (int j = 0; j < nJ; j++)
      for (int i = 0; i < nI; i++)
        this.wIJ[j][i] = w[j][i];
  }

  public void setWeightsJK(double[][] w) {
    if (w.length != nK || w[0].length != nJ)
      throw new IllegalArgumentException("Dimensiones de wJK no coinciden");
    for (int k = 0; k < nK; k++)
      for (int j = 0; j < nJ; j++)
        this.wJK[k][j] = w[k][j];
  }

  public static class ForwardResult {
    public final double[] aJ; // activaciones ocultas
    public final double[] aK; // salidas
    public ForwardResult(double[] aJ, double[] aK) { this.aJ = aJ; this.aK = aK; }
  }

  public ForwardResult forward(double[] x) {
    // aJ = sigmoid(wIJ * x)
    double[] aJ = new double[nJ];
    for (int j = 0; j < nJ; j++) {
      double sum = 0.0;
      for (int i = 0; i < nI; i++) sum += wIJ[j][i] * x[i];
      aJ[j] = sigmoid(sum);
    }
    // aK = sigmoid(wJK * aJ)
    double[] aK = new double[nK];
    for (int k = 0; k < nK; k++) {
      double sum = 0.0;
      for (int j = 0; j < nJ; j++) sum += wJK[k][j] * aJ[j];
      aK[k] = sigmoid(sum);
    }
    return new ForwardResult(aJ, aK);
  }

  public static class BackpropResult {
    public final double[] deltaK;
    public final double[] deltaJ;
    public final double[][] deltaW_JK; // [k][j]
    public final double[][] deltaW_IJ; // [j][i]
    public BackpropResult(double[] deltaK, double[] deltaJ, double[][] deltaW_JK, double[][] deltaW_IJ) {
      this.deltaK = deltaK; this.deltaJ = deltaJ; this.deltaW_JK = deltaW_JK; this.deltaW_IJ = deltaW_IJ;
    }
  }

  public BackpropResult backprop(double[] x, ForwardResult fr, double[] yTarget) {
    // Error en salida: e = y - aK
    double[] e = new double[nK];
    for (int k = 0; k < nK; k++) e[k] = yTarget[k] - fr.aK[k];

    // Gradiente salida: deltaK = aK*(1-aK)*e
    double[] deltaK = new double[nK];
    for (int k = 0; k < nK; k++) deltaK[k] = fr.aK[k] * (1 - fr.aK[k]) * e[k];

    // Gradiente oculta: deltaJ = aJ*(1-aJ) * (WJK^T * deltaK)
    double[] deltaJ = new double[nJ];
    for (int j = 0; j < nJ; j++) {
      double sum = 0.0;
      for (int k = 0; k < nK; k++) sum += wJK[k][j] * deltaK[k];
      deltaJ[j] = fr.aJ[j] * (1 - fr.aJ[j]) * sum;
    }

    // Deltas de pesos
    double[][] deltaW_JK = new double[nK][nJ];
    for (int k = 0; k < nK; k++)
      for (int j = 0; j < nJ; j++)
        deltaW_JK[k][j] = alpha * fr.aJ[j] * deltaK[k];

    double[][] deltaW_IJ = new double[nJ][nI];
    for (int j = 0; j < nJ; j++)
      for (int i = 0; i < nI; i++)
        deltaW_IJ[j][i] = alpha * x[i] * deltaJ[j];

    return new BackpropResult(deltaK, deltaJ, deltaW_JK, deltaW_IJ);
  }

  public void applyDeltas(BackpropResult br) {
    // w = w + Δw
    for (int k = 0; k < nK; k++)
      for (int j = 0; j < nJ; j++)
        wJK[k][j] += br.deltaW_JK[k][j];

    for (int j = 0; j < nJ; j++)
      for (int i = 0; i < nI; i++)
        wIJ[j][i] += br.deltaW_IJ[j][i];
  }

  public void printWeights() {
    System.out.printf("Pesos I->J (wIJ) [%d x %d]:\n", nJ, nI);
    for (int j = 0; j < nJ; j++) {
      System.out.printf("  j=%d: [", j);
      for (int i = 0; i < nI; i++) System.out.printf("%s%.6f", (i==0?"":", "), wIJ[j][i]);
      System.out.println("]");
    }
    System.out.printf("Pesos J->K (wJK) [%d x %d]:\n", nK, nJ);
    for (int k = 0; k < nK; k++) {
      System.out.printf("  k=%d: [", k);
      for (int j = 0; j < nJ; j++) System.out.printf("%s%.6f", (j==0?"":", "), wJK[k][j]);
      System.out.println("]");
    }
  }

  private static double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }
}
