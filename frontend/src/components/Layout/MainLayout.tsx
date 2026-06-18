import React from 'react';
import { Sidebar } from './Sidebar';

interface MainLayoutProps {
  children: React.ReactNode;
  currentView: string;
  onNavigate: (view: string) => void;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children, currentView, onNavigate }) => {
  const [isCollapsed, setIsCollapsed] = React.useState(false);

  return (
    <div className="flex h-screen w-full bg-slate-950 overflow-hidden">
      <Sidebar 
        currentView={currentView} 
        onNavigate={onNavigate}
        isCollapsed={isCollapsed}
        toggleSidebar={() => setIsCollapsed(!isCollapsed)}
      />
      <main className="flex-1 h-full overflow-hidden relative flex flex-col">
        {/* Top Bar / Breadcrumbs could go here */}
        <div className="flex-1 overflow-auto relative">
          {children}
        </div>
      </main>
    </div>
  );
};
