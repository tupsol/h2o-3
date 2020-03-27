package hex.tree.xgboost.predict;

import hex.DataInfo;
import hex.Model;
import hex.genmodel.algos.xgboost.XGBoostMojoModel;
import hex.tree.xgboost.BoosterParms;
import hex.tree.xgboost.XGBoostModel;
import hex.tree.xgboost.XGBoostOutput;
import hex.tree.xgboost.XGBoostUtils;
import hex.tree.xgboost.util.BoosterHelper;
import ml.dmlc.xgboost4j.java.*;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.util.Log;

import java.util.HashMap;

public class XGBoostNativeBigScoreChunkPredict implements XGBoostPredictContrib, Model.BigScoreChunkPredict {

  private final double _threshold;
  private final int _responseIndex;
  private final int _offsetIndex;

  private final XGBoostModelInfo _modelInfo;
  private final XGBoostModel.XGBoostParameters _parms;
  private final DataInfo _dataInfo;
  private final BoosterParms _boosterParms;
  private final XGBoostOutput _output;

  private final float[][] _preds;

  public XGBoostNativeBigScoreChunkPredict(
      XGBoostModelInfo modelInfo,
      XGBoostModel.XGBoostParameters parms,
      DataInfo di,
      BoosterParms boosterParms,
      double threshold,
      XGBoostOutput output,
      Frame fr,
      Chunk[] chks
  ) {
    _modelInfo = modelInfo;
    _parms = parms;
    _dataInfo = di;
    _boosterParms = boosterParms;
    _threshold = threshold;
    _output = output;
    
    _responseIndex = fr.find(_parms._response_column);
    _offsetIndex = fr.find(_parms._offset_column);
    _preds = scoreChunk(chks, XGBoostPredict.OutputType.PREDICT);
  }

  @Override
  public double[] score0(Chunk[] chks, double offset, int row_in_chunk, double[] tmp, double[] preds) {
    for (int i = 0; i < tmp.length; i++) {
      tmp[i] = chks[i].atd(row_in_chunk);
    }
    return XGBoostMojoModel.toPreds(tmp, _preds[row_in_chunk], preds, _output.nclasses(), null, _threshold);
  }

  @Override
  public float[][] predictContrib(Chunk[] cs) {
    return scoreChunk(cs, OutputType.PREDICT_CONTRIB_APPROX);
  }

  @Override
  public float[][] predict(Chunk[] cs) {
    return scoreChunk(cs, OutputType.PREDICT);
  }

  private float[][] scoreChunk(final Chunk[] cs, final XGBoostPredict.OutputType outputType) {
    DMatrix data = null;
    Booster booster = null;
    try {
      // Rabit has to be initialized as parts of booster.predict() are using Rabit
      // This might be fixed in future versions of XGBoost
      Rabit.init(new HashMap<>());
      data = XGBoostUtils.convertChunksToDMatrix(
          _dataInfo,
          cs,
          _responseIndex,
          _output._sparse,
          _offsetIndex
      );
      // No local chunks for this frame
      if (data.rowNum() == 0) {
        return null;
      }
      // Initialize Booster
      booster = _modelInfo.deserializeBooster();
      booster.setParams(_boosterParms.get());
      int treeLimit = 0;
      if (_parms._booster == XGBoostModel.XGBoostParameters.Booster.dart) {
        // DART with treeLimit=0 returns non-deterministic random predictions
        treeLimit = _parms._ntrees;
      }

      // Predict
      float[][] preds;
      switch (outputType) {
        case PREDICT:
          preds = booster.predict(data, false, treeLimit);
          break;
        case PREDICT_CONTRIB_APPROX:
          preds = booster.predictContrib(data, treeLimit);
          break;
        default:
          throw new UnsupportedOperationException("Unsupported output type: " + outputType);
      }
      return preds == null ? new float[0][] : preds;
    } catch (XGBoostError xgBoostError) {
      throw new IllegalStateException("Failed to score with XGBoost.", xgBoostError);
    } finally {
      BoosterHelper.dispose(booster, data);
      try {
        Rabit.shutdown();
      } catch (XGBoostError xgBoostError) {
        Log.err("Failed Rabit shutdown. A hanging RabitTracker task might be present on the driver node.", xgBoostError);
      }
    }
  }

  @Override
  public void close() {
  }

}
