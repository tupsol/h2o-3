package hex.gam;

import hex.CreateFrame;
import hex.glm.GLMModel;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import water.DKV;
import water.Scope;
import water.fvec.Frame;
import water.runner.CloudSize;
import water.runner.H2ORunner;

import java.util.ArrayList;

import static hex.gam.GamTestPiping.getModel;
import static hex.gam.GamTestPiping.massageFrame;
import static hex.glm.GLMModel.GLMParameters.Family.*;
import static water.TestUtil.parse_test_file;

@RunWith(H2ORunner.class)
@CloudSize(1)
public class GamMojoModelTest {
  public static final double _tol = 1e-6;
  
  // test and make sure the h2opredict, mojo predict agrees with binomial dataset that includes
  // both enum and numerical datasets for the binomial family
  @Test
  public void testBinomialPredMojo() {
    Scope.enter();
    try {
      // test for binomial
      String[] ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
              "C15", "C16", "C17", "C18", "C19", "C20"};
      String[] gamCols = new String[]{"C11", "C12", "C13"};
      Frame trainBinomial = Scope.track(massageFrame(parse_test_file("smalldata/glm_test/binomial_20_cols_10KRows.csv"),
              binomial));
      DKV.put(trainBinomial);
      GAMModel binomialModel = getModel(binomial,
              parse_test_file("smalldata/glm_test/binomial_20_cols_10KRows.csv"), "C21",
              gamCols, ignoredCols, new int[]{5, 5, 5}, new int[]{0, 0, 0}, false, true,
              new double[]{1, 1, 1}, new double[]{0, 0, 0}, new double[]{0, 0, 0}, true, null,
              null, false);
      Scope.track_generic(binomialModel);
      binomialModel._output._training_metrics = null; // force prediction threshold of 0.5
      Frame predictBinomial = Scope.track(binomialModel.score(trainBinomial));
      Assert.assertTrue(binomialModel.testJavaScoring(trainBinomial, predictBinomial, _tol));
    } finally {
      Scope.exit();
    }
  }

  // test and make sure the h2opredict, mojo predict agrees with gaussian dataset that includes
  // both enum and numerical datasets for the gaussian family
  @Test
  public void testGaussianPredMojo() {
    Scope.enter();
    try {
      String[] ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
              "C15", "C16", "C17", "C18", "C19", "C20"};
      String[] gamCols = new String[]{"C11", "C12", "C13"};
      Frame trainGaussian = Scope.track(massageFrame(parse_test_file("smalldata/glm_test/gaussian_20cols_10000Rows.csv"), gaussian));
      DKV.put(trainGaussian);
      GAMModel gaussianmodel = getModel(gaussian,
              parse_test_file("smalldata/glm_test/gaussian_20cols_10000Rows.csv"), "C21",
              gamCols, ignoredCols, new int[]{5, 5, 5}, new int[]{0, 0, 0}, false, true,
              new double[]{1, 1, 1}, new double[]{0, 0, 0}, new double[]{0, 0, 0}, true, null,null, true);
      Scope.track_generic(gaussianmodel);
      Frame predictGaussian = Scope.track(gaussianmodel.score(trainGaussian));
      Frame predictG = new Frame(predictGaussian.vec(0));
      Scope.track(predictG);

      Assert.assertTrue(gaussianmodel.testJavaScoring(trainGaussian, predictG, _tol)); // compare scoring result with mojo
    } finally {
      Scope.exit();
    }
  }

  // test and make sure the h2opredict, mojo predict agrees with multinomial dataset that includes
  // both enum and numerical datasets for the multinomial family
  @Test
  public void testMultinomialModelMojo() {
    Scope.enter();
    try {
      // multinomial
      String[] ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[] gamCols = new String[]{"C6", "C7", "C8"};
      Frame trainMultinomial = Scope.track(massageFrame(parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"), multinomial));
      DKV.put(trainMultinomial);
      GAMModel multinomialModel = getModel(multinomial,
              parse_test_file("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"),
              "C11", gamCols, ignoredCols, new int[]{5, 5, 5}, new int[]{0, 0, 0}, false,
              true, new double[]{1, 1, 1}, new double[]{0, 0, 0}, new double[]{0, 0, 0},
              true, null,null, false);
      Scope.track_generic(multinomialModel);
      Frame predictMult = Scope.track(multinomialModel.score(trainMultinomial));
      Assert.assertTrue(multinomialModel.testJavaScoring(trainMultinomial, predictMult, _tol)); // compare scoring result with mojo
    } finally {
      Scope.exit();
    }
  }
  
  
  public GAMModel.GAMParameters buildGamParams(Frame train, GLMModel.GLMParameters.Family fam) {
    GAMModel.GAMParameters paramsO = new GAMModel.GAMParameters();
    paramsO._train = train._key;
    paramsO._lambda_search = false;
    paramsO._response_column = "response";
    paramsO._lambda = new double[]{0};
    paramsO._alpha = new double[]{0.001};  // l1pen
    paramsO._objective_epsilon = 1e-6;
    paramsO._beta_epsilon = 1e-4;
    paramsO._standardize = false;
    paramsO._family = fam;
    paramsO._gam_columns =  chooseGamColumns(train, 3);
    return paramsO;
  }

  public String[] chooseGamColumns(Frame trainF, int maxGamCols) {
    int gamCount=0;
    ArrayList<String> numericCols = new ArrayList<>();
    String[] colNames = trainF.names();
    for (String cnames : colNames) {
      if (trainF.vec(cnames).isNumeric() && !trainF.vec(cnames).isInt()) {
        numericCols.add(cnames);
        gamCount++;
      }
      if (gamCount >= maxGamCols)
        break;
    }
    String[] gam_columns = new String[numericCols.size()];
    return numericCols.toArray(gam_columns);
  }
  
  public Frame createTrainTestFrame(int responseFactor) {
    CreateFrame cf = new CreateFrame();
    int numRows = 18888;
    int numCols = 18;
    cf.rows= numRows;
    cf.cols = numCols;
    cf.factors=10;
    cf.has_response=true;
    cf.response_factors = responseFactor; // 1 for real-value response
    cf.positive_response=true;
    cf.missing_fraction = 0;
    cf.seed = 12345;
    System.out.println("Createframe parameters: rows: "+numRows+" cols:"+numCols+" seed: "+cf.seed);
    return cf.execImpl().get();
  }
}
