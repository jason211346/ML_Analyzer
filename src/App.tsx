import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import toast from 'react-hot-toast';
import Sidebar from './components/Sidebar';
import FileUploader from './components/FileUploader';
import DataTable from './components/DataTable';
import AnalysisPanel from './components/AnalysisPanel';
import { User, ExcelData } from './types';
import { 
  createUserFolder, 
  deleteUserFolder, 
  getUsersFromStorage,
  saveFileForUser,
  getFilesForUser,
  deleteFileForUser
} from './utils/fileSystem';
import { 
  parseFile, 
  excelToBase64,
  parseBase64Excel
} from './utils/excelParser';
import { generateUUID } from './utils/uuid';
import { FileText, Database, BarChart2, Trash2 } from 'lucide-react';

function App() {
  const [users, setUsers] = useState<User[]>([]);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [excelData, setExcelData] = useState<ExcelData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [userFiles, setUserFiles] = useState<any[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<'data' | 'analysis'>('data');

  // Load users from localStorage on initial render
  useEffect(() => {
    const storedUsers = getUsersFromStorage();
    setUsers(storedUsers);
  }, []);

  // Load user files when a user is selected
  useEffect(() => {
    if (selectedUser) {
      const files = getFilesForUser(selectedUser.id);
      setUserFiles(files);
      
      // If there's a selected file, load its data
      if (selectedFile) {
        const file = files.find((f: any) => f.name === selectedFile);
        if (file) {
          loadFileData(file.data);
        }
      } else {
        setExcelData(null);
      }
    } else {
      setUserFiles([]);
      setSelectedFile(null);
      setExcelData(null);
    }
  }, [selectedUser, selectedFile]);

  const handleAddUser = (name: string) => {
    const newUser: User = {
      id: generateUUID(),
      name,
      folderPath: `/users/${name.toLowerCase().replace(/\s+/g, '_')}`,
      createdAt: new Date().toISOString()
    };
    
    const success = createUserFolder(newUser);
    
    if (success) {
      setUsers(prev => [...prev, newUser]);
      toast.success(`使用者 "${name}" 已建立`);
    } else {
      toast.error('建立使用者失敗');
    }
  };

  const handleDeleteUser = (userId: string) => {
    const user = users.find(u => u.id === userId);
    
    if (!user) {
      toast.error('找不到使用者');
      return;
    }
    
    const success = deleteUserFolder(userId);
    
    if (success) {
      setUsers(prev => prev.filter(u => u.id !== userId));
      
      if (selectedUser?.id === userId) {
        setSelectedUser(null);
        setExcelData(null);
      }
      
      toast.success(`使用者 "${user.name}" 已刪除`);
    } else {
      toast.error('刪除使用者失敗');
    }
  };

  const handleSelectUser = (user: User) => {
    setSelectedUser(user);
    setSelectedFile(null);
    setExcelData(null);
  };

  const handleFileUpload = async (file: File) => {
    if (!selectedUser) {
      toast.error('請先選擇使用者');
      return;
    }
    
    setIsLoading(true);
    
    try {
      // Parse file (Excel or CSV)
      const data = await parseFile(file);
      
      // Convert file to base64 for storage
      const base64Data = await excelToBase64(file);
      
      // Save file to user's folder
      const success = saveFileForUser(selectedUser.id, file.name, base64Data);
      
      if (success) {
        setExcelData(data);
        setSelectedFile(file.name);
        
        // Refresh user files
        const files = getFilesForUser(selectedUser.id);
        setUserFiles(files);
        
        const fileType = file.name.endsWith('.csv') ? 'CSV' : 'Excel';
        toast.success(`${fileType} 檔案 "${file.name}" 已上傳`);
      } else {
        toast.error('儲存檔案失敗');
      }
    } catch (error) {
      console.error('File upload error:', error);
      toast.error('處理檔案時發生錯誤');
    } finally {
      setIsLoading(false);
    }
  };

  const loadFileData = async (base64Data: string) => {
    setIsLoading(true);
    
    try {
      const data = await parseBase64Excel(base64Data);
      setExcelData(data);
    } catch (error) {
      console.error('Error loading file data:', error);
      toast.error('載入檔案資料失敗');
      setExcelData(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectFile = (fileName: string) => {
    setSelectedFile(fileName);
    
    const file = userFiles.find((f: any) => f.name === fileName);
    if (file) {
      loadFileData(file.data);
    }
  };

  const handleDeleteFile = (fileName: string) => {
    if (!selectedUser) {
      toast.error('請先選擇使用者');
      return;
    }
    
    if (confirm(`確定要刪除檔案 "${fileName}" 嗎？`)) {
      const success = deleteFileForUser(selectedUser.id, fileName);
      
      if (success) {
        // Refresh user files
        const files = getFilesForUser(selectedUser.id);
        setUserFiles(files);
        
        // If the deleted file was selected, clear selection
        if (selectedFile === fileName) {
          setSelectedFile(null);
          setExcelData(null);
        }
        
        toast.success(`檔案 "${fileName}" 已刪除`);
      } else {
        toast.error('刪除檔案失敗');
      }
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      <Toaster position="top-right" />
      
      <Sidebar
        users={users}
        selectedUser={selectedUser}
        onSelectUser={handleSelectUser}
        onAddUser={handleAddUser}
        onDeleteUser={handleDeleteUser}
      />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white shadow-sm p-4">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-800">Excel 資料分析系統</h1>
            {selectedUser && (
              <div className="text-sm text-gray-600">
                目前使用者: <span className="font-semibold">{selectedUser.name}</span>
              </div>
            )}
          </div>
        </header>
        
        <main className="flex-1 overflow-y-auto p-6">
          {!selectedUser ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <Database size={64} />
              <p className="mt-4 text-xl">請從左側選單選擇使用者</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-6">
              {userFiles.length > 0 && (
                <div className="bg-white rounded-lg shadow p-6">
                  <h2 className="text-lg font-semibold mb-4">已上傳的檔案</h2>
                  <div className="flex flex-wrap gap-3">
                    {userFiles.map((file: any, index: number) => (
                      <div 
                        key={index}
                        className={`flex items-center p-3 border rounded-lg ${
                          selectedFile === file.name 
                            ? 'border-blue-500 bg-blue-50' 
                            : 'border-gray-200 hover:bg-gray-50'
                        }`}
                      >
                        <div 
                          className="flex items-center cursor-pointer"
                          onClick={() => handleSelectFile(file.name)}
                        >
                          <FileText size={20} className="text-blue-500 mr-2" />
                          <div>
                            <div className="font-medium">{file.name}</div>
                            <div className="text-xs text-gray-500">
                              {new Date(file.uploadedAt).toLocaleString()}
                            </div>
                          </div>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteFile(file.name);
                          }}
                          className="ml-3 p-1 text-red-500 hover:bg-red-50 rounded"
                          title="刪除檔案"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold mb-4">上傳資料檔案</h2>
                <FileUploader onFileUpload={handleFileUpload} isLoading={isLoading} />
              </div>
              
              {excelData && (
                <div className="bg-white rounded-lg shadow">
                  <div className="border-b flex">
                    <button
                      className={`px-4 py-3 flex items-center ${
                        activeView === 'data'
                          ? 'border-b-2 border-blue-500 text-blue-600'
                          : 'text-gray-600 hover:text-gray-800'
                      }`}
                      onClick={() => setActiveView('data')}
                    >
                      <Database size={18} className="mr-2" />
                      資料表
                    </button>
                    <button
                      className={`px-4 py-3 flex items-center ${
                        activeView === 'analysis'
                          ? 'border-b-2 border-blue-500 text-blue-600'
                          : 'text-gray-600 hover:text-gray-800'
                      }`}
                      onClick={() => setActiveView('analysis')}
                    >
                      <BarChart2 size={18} className="mr-2" />
                      資料分析
                    </button>
                  </div>
                  
                  <div className="p-6">
                    {activeView === 'data' ? (
                      <DataTable data={excelData} />
                    ) : (
                      <AnalysisPanel data={excelData} />
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;