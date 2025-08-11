import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './components/HomePage';
import Dashboard from './components/Dashboard';
import Particles from './components/Particles';
import FileUpload from './components/FileUpload';
import DatabaseMonitor from './components/DatabaseMonitor'; // Import the new component

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/home" replace />} />
      <Route path="/home" element={<HomePage />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/about" element={<Particles />} />
      <Route path="/upload" element={<FileUpload />} />
      <Route path="/database" element={<DatabaseMonitor />} /> {/* New database monitor route */}
    </Routes>
  );
}

export default App;