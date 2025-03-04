import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Plus, Trash2, Users } from 'lucide-react';
import { User } from '../types';
import toast from 'react-hot-toast';

interface SidebarProps {
  users: User[];
  selectedUser: User | null;
  onSelectUser: (user: User) => void;
  onAddUser: (name: string) => void;
  onDeleteUser: (userId: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  users,
  selectedUser,
  onSelectUser,
  onAddUser,
  onDeleteUser
}) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [newUserName, setNewUserName] = useState('');
  const [showAddUserForm, setShowAddUserForm] = useState(false);

  const handleAddUser = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!newUserName.trim()) {
      toast.error('使用者名稱不能為空');
      return;
    }
    
    onAddUser(newUserName);
    setNewUserName('');
    setShowAddUserForm(false);
  };

  const handleDeleteUser = (userId: string) => {
    if (confirm('確定要刪除此使用者嗎？所有相關檔案將會被移除。')) {
      onDeleteUser(userId);
    }
  };

  return (
    <div className="w-64 bg-gray-800 text-white h-full flex flex-col">
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-xl font-semibold flex items-center">
          <Users className="mr-2" size={20} />
          使用者管理
        </h2>
      </div>
      
      <div className="p-4">
        <div 
          className="flex items-center justify-between p-2 bg-gray-700 rounded cursor-pointer"
          onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        >
          <span>{selectedUser ? selectedUser.name : '選擇使用者'}</span>
          {isDropdownOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </div>
        
        {isDropdownOpen && (
          <div className="mt-1 bg-gray-700 rounded overflow-hidden">
            {users.length === 0 ? (
              <div className="p-2 text-gray-400 text-sm">尚無使用者</div>
            ) : (
              users.map(user => (
                <div 
                  key={user.id}
                  className={`p-2 flex justify-between items-center cursor-pointer hover:bg-gray-600 ${
                    selectedUser?.id === user.id ? 'bg-gray-600' : ''
                  }`}
                >
                  <span 
                    className="flex-1"
                    onClick={() => {
                      onSelectUser(user);
                      setIsDropdownOpen(false);
                    }}
                  >
                    {user.name}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteUser(user.id);
                    }}
                    className="text-red-400 hover:text-red-300"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>
      
      <div className="p-4">
        {showAddUserForm ? (
          <form onSubmit={handleAddUser} className="space-y-2">
            <input
              type="text"
              value={newUserName}
              onChange={(e) => setNewUserName(e.target.value)}
              placeholder="輸入使用者名稱"
              className="w-full p-2 rounded bg-gray-700 text-white"
              autoFocus
            />
            <div className="flex space-x-2">
              <button
                type="submit"
                className="px-3 py-1 bg-blue-600 rounded hover:bg-blue-500 text-sm"
              >
                新增
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowAddUserForm(false);
                  setNewUserName('');
                }}
                className="px-3 py-1 bg-gray-600 rounded hover:bg-gray-500 text-sm"
              >
                取消
              </button>
            </div>
          </form>
        ) : (
          <button
            onClick={() => setShowAddUserForm(true)}
            className="flex items-center justify-center w-full p-2 bg-blue-600 rounded hover:bg-blue-500"
          >
            <Plus size={18} className="mr-1" />
            新增使用者
          </button>
        )}
      </div>
      
      <div className="mt-auto p-4 text-xs text-gray-400 border-t border-gray-700">
        使用者資料將儲存於本地端
      </div>
    </div>
  );
};

export default Sidebar;