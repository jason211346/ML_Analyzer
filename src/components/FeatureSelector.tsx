import React from 'react';
import { ExcelData, FeatureSelection } from '../types';

interface FeatureSelectorProps {
  data: ExcelData | null;
  featureSelection: FeatureSelection;
  onFeatureSelectionChange: (selection: FeatureSelection) => void;
}

const FeatureSelector: React.FC<FeatureSelectorProps> = ({
  data,
  featureSelection,
  onFeatureSelectionChange
}) => {
  if (!data) {
    return null;
  }

  const handleInputFeatureChange = (feature: string) => {
    const updatedInputFeatures = featureSelection.inputFeatures.includes(feature)
      ? featureSelection.inputFeatures.filter(f => f !== feature)
      : [...featureSelection.inputFeatures, feature];
    
    onFeatureSelectionChange({
      ...featureSelection,
      inputFeatures: updatedInputFeatures
    });
  };

  const handleTargetFeatureChange = (feature: string) => {
    const updatedTargetFeatures = featureSelection.targetFeatures.includes(feature)
      ? featureSelection.targetFeatures.filter(f => f !== feature)
      : [...featureSelection.targetFeatures, feature];
    
    onFeatureSelectionChange({
      ...featureSelection,
      targetFeatures: updatedTargetFeatures
    });
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h2 className="text-xl font-semibold mb-4">特徵選擇</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-lg font-medium mb-2">輸入特徵 (X)</h3>
          <p className="text-sm text-gray-500 mb-3">選擇用於預測的輸入變數</p>
          <div className="max-h-60 overflow-y-auto border rounded-lg p-3">
            {data.headers.map((header, index) => (
              <div key={`input-${index}`} className="flex items-center mb-2">
                <input
                  type="checkbox"
                  id={`input-${index}`}
                  checked={featureSelection.inputFeatures.includes(header)}
                  onChange={() => handleInputFeatureChange(header)}
                  className="mr-2 h-4 w-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <label htmlFor={`input-${index}`} className="text-sm">
                  {header}
                </label>
              </div>
            ))}
          </div>
          <div className="mt-2 text-sm text-gray-500">
            已選擇 {featureSelection.inputFeatures.length} 個輸入特徵
          </div>
        </div>
        
        <div>
          <h3 className="text-lg font-medium mb-2">預測目標 (Y)</h3>
          <p className="text-sm text-gray-500 mb-3">選擇要預測的目標變數</p>
          <div className="max-h-60 overflow-y-auto border rounded-lg p-3">
            {data.headers.map((header, index) => (
              <div key={`target-${index}`} className="flex items-center mb-2">
                <input
                  type="checkbox"
                  id={`target-${index}`}
                  checked={featureSelection.targetFeatures.includes(header)}
                  onChange={() => handleTargetFeatureChange(header)}
                  className="mr-2 h-4 w-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <label htmlFor={`target-${index}`} className="text-sm">
                  {header}
                </label>
              </div>
            ))}
          </div>
          <div className="mt-2 text-sm text-gray-500">
            已選擇 {featureSelection.targetFeatures.length} 個預測目標
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeatureSelector;