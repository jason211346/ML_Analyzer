import React, { useState } from 'react';
import { ExcelData, FeatureSelection, ModelResults as ModelResultsType } from '../types';
import FeatureSelector from './FeatureSelector';
import ModelTrainer from './ModelTrainer';
import ModelResults from './ModelResults';
import { LineChart, BarChart2, BrainCircuit } from 'lucide-react';

interface AnalysisPanelProps {
  data: ExcelData | null;
}

const AnalysisPanel: React.FC<AnalysisPanelProps> = ({ data }) => {
  const [featureSelection, setFeatureSelection] = useState<FeatureSelection>({
    inputFeatures: [],
    targetFeatures: []
  });
  
  const [modelResults, setModelResults] = useState<ModelResultsType | null>(null);
  const [activeTab, setActiveTab] = useState<'features' | 'training' | 'results'>('features');

  const handleFeatureSelectionChange = (selection: FeatureSelection) => {
    setFeatureSelection(selection);
  };

  const handleModelTrained = (results: ModelResultsType) => {
    setModelResults(results);
    setActiveTab('results');
  };

  if (!data) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="text-center p-8 text-gray-500">
          請先上傳Excel檔案以進行資料分析
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow">
        <div className="flex border-b">
          <button
            className={`flex items-center px-4 py-3 ${
              activeTab === 'features'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveTab('features')}
          >
            <BarChart2 size={18} className="mr-2" />
            特徵選擇
          </button>
          <button
            className={`flex items-center px-4 py-3 ${
              activeTab === 'training'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveTab('training')}
          >
            <BrainCircuit size={18} className="mr-2" />
            模型訓練
          </button>
          <button
            className={`flex items-center px-4 py-3 ${
              activeTab === 'results'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-600 hover:text-gray-800'
            } ${!modelResults ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={() => modelResults && setActiveTab('results')}
            disabled={!modelResults}
          >
            <LineChart size={18} className="mr-2" />
            分析結果
          </button>
        </div>
        
        <div className="p-4">
          {activeTab === 'features' && (
            <FeatureSelector
              data={data}
              featureSelection={featureSelection}
              onFeatureSelectionChange={handleFeatureSelectionChange}
            />
          )}
          
          {activeTab === 'training' && (
            <ModelTrainer
              data={data}
              featureSelection={featureSelection}
              onModelTrained={handleModelTrained}
            />
          )}
          
          {activeTab === 'results' && (
            <ModelResults results={modelResults} />
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisPanel;