package com.app;

import com.app.ml.NeuralNetwork;
import java.util.Locale;
import java.util.Scanner;

public class MainApp {

  static double readDouble(Scanner sc, String prompt) {
    while (true) {
      System.out.print(prompt);
      String s = sc.nextLine().trim();
      try { return Double.parseDouble(s); } catch (Exception e) { System.out.println("  -> número inválido"); }
    }
  }

  static int readInt(Scanner sc, String prompt) {
    while (true) {
      System.out.print(prompt);
      String s = sc.nextLine().trim();
      try { return Integer.parseInt(s); } catch (Exception e) { System.out.println("  -> entero inválido"); }
    }
  }

  public static void main(String[] args) {
    Locale.setDefault(Locale.US);
    Scanner sc = new Scanner(System.in);

    System.out.println("=== Backpropagation (consola) ===");
    int nI = readInt(sc, "Cantidad de neuronas de entrada (I): ");
    int nJ = readInt(sc, "Cantidad de neuronas en capa oculta (J): ");
    int nK = readInt(sc, "Cantidad de neuronas de salida (K): ");

    double alpha = readDouble(sc, "Tasa de aprendizaje (α): ");
    int iters = readInt(sc, "Número de iteraciones (1-10): ");
    if (iters < 1) iters = 1;
    if (iters > 10) iters = 10;

    // Etiquetas de neuronas
    String[] labX = buildInputLabels(nI);                // x1..xN
    String[] labJ = buildHiddenLabels(nI, nJ);           // y(nI+1)..
    String[] labK = buildOutputLabels(nI, nJ, nK);       // y(nI+nJ+1)..

    // Entradas
    double[] x = new double[nI];
    System.out.println("\nIntroduce los valores de entrada I (" + String.join(", ", labX) + "):");
    for (int i = 0; i < nI; i++) x[i] = readDouble(sc, String.format("  %s = ", labX[i]));

    // Salidas esperadas
    double[] y = new double[nK];
    System.out.println("\nIntroduce las salidas esperadas K (" + String.join(", ", labK) + "):");
    for (int k = 0; k < nK; k++) y[k] = readDouble(sc, String.format("  %s* = ", labK[k]));

    // Pesos I->J
    double[][] wIJ = new double[nJ][nI];
    System.out.println("\nIntroduce los pesos I->J (de " + String.join(", ", labX) + " a " + String.join(", ", labJ) + "):");
    for (int j = 0; j < nJ; j++) {
      for (int i = 0; i < nI; i++) {
        wIJ[j][i] = readDouble(sc, String.format("  w(%s→%s) = ", labX[i], labJ[j]));
      }
    }

    // Pesos J->K
    double[][] wJK = new double[nK][nJ];
    System.out.println("\nIntroduce los pesos J->K (de " + String.join(", ", labJ) + " a " + String.join(", ", labK) + "):");
    for (int k = 0; k < nK; k++) {
      for (int j = 0; j < nJ; j++) {
        wJK[k][j] = readDouble(sc, String.format("  w(%s→%s) = ", labJ[j], labK[k]));
      }
    }

    NeuralNetwork nn = new NeuralNetwork(nI, nJ, nK, alpha);
    nn.setWeightsIJ(wIJ);
    nn.setWeightsJK(wJK);

    // Históricos
    double[][] histAJ = new double[iters][nJ]; // activaciones ocultas
    double[][] histAK = new double[iters][nK]; // activaciones salida
    double[][] histE  = new double[iters][nK]; // errores
    double[]  histMSE = new double[iters];     // MSE (promedio de e^2)

    for (int it = 0; it < iters; it++) {
      NeuralNetwork.ForwardResult fr = nn.forward(x);

      // Error y métricas
      double[] e = new double[nK];
      double sumSq = 0.0;
      for (int k = 0; k < nK; k++) {
        e[k] = y[k] - fr.aK[k];
        sumSq += e[k] * e[k];
      }
      double mse = sumSq / nK;  // MSE mostrado como métrica

      // Guardar históricos
      System.arraycopy(fr.aJ, 0, histAJ[it], 0, nJ);
      System.arraycopy(fr.aK, 0, histAK[it], 0, nK);
      System.arraycopy(e,     0, histE[it],  0, nK);
      histMSE[it] = mse;

      // Backprop + update
      NeuralNetwork.BackpropResult br = nn.backprop(x, fr, y);
      nn.applyDeltas(br);
    }

    // Tablas
    printPerIterationTable(histAJ, histAK, histE, histMSE, labJ, labK);
    System.out.println("\n--- PESOS FINALES ---");
    nn.printWeights();

    System.out.println("\nListo. (Entrenamiento de " + iters + " iteración/iteraciones)");
  }

  // ===== etiquetas =====
  static String[] buildInputLabels(int nI) {
    String[] lab = new String[nI];
    for (int i = 0; i < nI; i++) lab[i] = "x" + (i + 1);
    return lab;
  }
  static String[] buildHiddenLabels(int nI, int nJ) {
    String[] lab = new String[nJ];
    for (int j = 0; j < nJ; j++) lab[j] = "y" + (nI + (j + 1));
    return lab;
  }
  static String[] buildOutputLabels(int nI, int nJ, int nK) {
    String[] lab = new String[nK];
    int start = nI + nJ + 1;
    for (int k = 0; k < nK; k++) lab[k] = "y" + (start + k);
    return lab;
  }

  // ===== impresión tablas =====
  static void printPerIterationTable(double[][] histAJ, double[][] histAK, double[][] histE, double[] histMSE,
                                     String[] labJ, String[] labK) {
    int iters = histAK.length;
    int nJ = labJ.length, nK = labK.length;

    System.out.println("\n================= TABLA POR ITERACIÓN =================");
    // Encabezado
    System.out.printf("%-6s", "iter");
    for (int j = 0; j < nJ; j++) System.out.printf("  a(%-5s) ", labJ[j]);  // ocultas
    for (int k = 0; k < nK; k++) System.out.printf("  a(%-5s) ", labK[k]);  // salidas
    for (int k = 0; k < nK; k++) System.out.printf("  e(%-5s) ", labK[k]);  // error por salida
    System.out.printf("  MSE%n");

    // Filas
    for (int it = 0; it < iters; it++) {
      System.out.printf("%-6d", (it + 1));
      for (int j = 0; j < nJ; j++) System.out.printf("  % -8.5f", histAJ[it][j]);
      for (int k = 0; k < nK; k++) System.out.printf("  % -8.5f", histAK[it][k]);
      for (int k = 0; k < nK; k++) System.out.printf("  % -8.5f", histE[it][k]);
      System.out.printf("  % -10.6f%n", histMSE[it]);
    }
    System.out.println("=======================================================");
  }
}
