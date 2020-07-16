package hex.genmodel.algos.gam;

import hex.genmodel.ConverterFactoryProvidingModel;
import hex.genmodel.GenModel;
import hex.genmodel.MojoModel;
import hex.genmodel.easy.CategoricalEncoder;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.RowToRawDataConverter;
import hex.genmodel.utils.ArrayUtils;
import hex.genmodel.utils.DistributionFamily;
import hex.genmodel.utils.LinkFunctionType;

import java.util.Map;

public abstract class GamMojoModelBase extends MojoModel implements ConverterFactoryProvidingModel {
  String _link;
  public LinkFunctionType _link_function;
  boolean _useAllFactorLevels;
  int _cats;
  int[] _catNAFills;
  int[] _catOffsets;
  int _nums;
  int _numsCenter;
  double[] _numNAFills;
  double[] _numNAFillsCenter;
  boolean _meanImputation;
  double[] _beta;
  double[] _beta_no_center;
  double[] _beta_center;
  double[][] _beta_multinomial;
  double[][] _beta_multinomial_no_center; // coefficients not centered for multinomial/ordinal
  double[][] _beta_multinomial_center; // coefficients not centered for multinomial/ordinal
  DistributionFamily _family;
  String[] _gam_columns;
  int _num_gam_columns;
  int[] _bs;
  int[] _num_knots;
  double[][] _knots;
  double[][][] _binvD;
  double[][][] _zTranspose;
  String[][] _gamColNames;  // expanded gam column names
  String[][] _gamColNamesCenter;
  String[] _names_no_centering; // column names of features with no centering
  int _totFeatureSize; // Gam Algo predictors numbers that include: predictors, expanded gam columns no centered
  int _betaSizePerClass;
  int _betaCenterSizePerClass;
  double _tweedieLinkPower;
  double[][] _basisVals; // store basis values array for each gam column
  double[][] _hj;   // difference between knot values
  int _numExpandedGamCols; // number of expanded gam columns
  int _lastClass;
  
  public int getTotFeatureSize() { return _totFeatureSize;}
  public String[] getNames() { return _names; }  // get column names centered
  
  GamMojoModelBase(String[] columns, String[][] domains, String responseColumn) {
    super(columns, domains, responseColumn);
  }

  @Override
  public double[] score0(double[] row, double[] preds) {
    if (_meanImputation)
      imputeMissingWithMeans(row);  // perform imputation for each row
    
    return gamScore0(row, preds);
  }
  
  void init() {
    _basisVals = new double[_gam_columns.length][];
    _hj = new double[_gam_columns.length][];
    for (int ind=0; ind < _num_gam_columns; ind++) {
      _basisVals[ind] = new double[_num_knots[ind]];
      _hj[ind] = ArrayUtils.eleDiff(_knots[ind]);
    }
    _lastClass = _nclasses - 1;
  }
  
  abstract double[] gamScore0(double[] row, double[] preds);
  
  private void imputeMissingWithMeans(double[] data) {
    for (int ind=0; ind < _cats; ind++)
      if (Double.isNaN(data[ind])) data[ind] = _catNAFills[ind];

    if (data.length == nfeatures()) { // using centered gam cols, nfeatures denotes centered gam columns
      for (int ind = 0; ind < _numsCenter; ind++)
        if (Double.isNaN(data[ind + _cats])) data[ind] = _numNAFillsCenter[ind];
    } else {
      for (int ind = 0; ind < _nums; ind++) {
        int colInd = ind+_cats;
        if (Double.isNaN(data[colInd])) 
          data[colInd] = _numNAFills[ind];
      }
    }
  }
  
  double evalLink(double val) {
    switch (_link_function) {
      case identity: return GenModel.GLM_identityInv(val);
      case logit: return GenModel.GLM_logitInv(val);
      case log: return GenModel.GLM_logInv(val);
      case inverse: return GenModel.GLM_inverseInv(val);
      case tweedie: return GenModel.GLM_tweedieInv(val, _tweedieLinkPower);
      default: throw new UnsupportedOperationException("Unexpected link function "+_link_function);
    }
  }

  // This method will read in categorical value and adjust for when useAllFactorLevels = true or false
  int readCatVal(double data, int dataIndex) {
    int ival = _useAllFactorLevels?((int) data):((int) data-1);
    double targetVal = _useAllFactorLevels?(data):(data-1);
    if (ival != targetVal) throw new IllegalArgumentException("categorical value out of range");
    if (ival < 0)
      return -1;
    ival += _catOffsets[dataIndex];
    return ival;
  }

  // this method will generate the beta*data+intercept
  double generateEta(double[] beta, double[] data) {
    double eta = 0.0;
    for (int i = 0; i < _catOffsets.length - 1; ++i) {  // take care of contribution from categorical columns
      int ival = readCatVal(data[i], i);
      if ((ival < _catOffsets[i + 1]) && (ival >= 0))
        eta += beta[ival];
    }

    int noff = _catOffsets[_cats] - _cats;
    for (int i = _cats; i < beta.length - 1 - noff; ++i)
      eta += beta[noff + i] * data[i];
    eta += beta[beta.length - 1]; // add intercept
    return eta;
  }
  
  // this method will add to each data row the expanded gam columns
  public void addExpandGamCols(double[] rawData, final RowData rowData) { // add all expanded gam columns here
    // add expanded gam columns to rowData
    int dataIndStart = _totFeatureSize-_numExpandedGamCols; // starting index to fill out the rawData
    for (int cind = 0; cind < _num_gam_columns; cind++) {
      if (_bs[cind]==0) { // to generate basis function values for cubic regression spline
        String dataString = ((String) rowData.get(_gam_columns[cind]));
        double gam_col_data = Double.NaN;
        if (dataString != null)
          gam_col_data = Double.parseDouble(dataString.trim());
        GamUtilsCubicRegression.expandOneGamCol(gam_col_data, _binvD[cind], _basisVals[cind], _hj[cind], _knots[cind]);
      } else {
        throw new IllegalArgumentException("spline type not implemented!");
      }
      System.arraycopy(_basisVals[cind], 0, rawData, dataIndStart, _num_knots[cind]); // copy expanded gam to rawData
      dataIndStart += _num_knots[cind];
    }
  }

  @Override
  public RowToRawDataConverter makeConverterFactory(Map<String, Integer> modelColumnNameToIndexMap,
                                                    Map<Integer, CategoricalEncoder> domainMap,
                                                    EasyPredictModelWrapper.ErrorConsumer errorConsumer,
                                                    EasyPredictModelWrapper.Config config) {
    return new GamRowToRawDataConverter(this, modelColumnNameToIndexMap, domainMap, errorConsumer, config);
  }
}
