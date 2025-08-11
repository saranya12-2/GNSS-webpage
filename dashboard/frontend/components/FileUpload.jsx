import React, { useState, useCallback } from 'react';
import axios from 'axios';
import styled, { createGlobalStyle } from 'styled-components';
import { Link } from 'react-router-dom';
import logo1 from '../assets/logo1.png';

const GlobalStyle = createGlobalStyle`
  :root {
    --dark-bg: #0a0a0f;
    --card-bg: rgba(25, 25, 40, 0.7);
    --sidebar-bg: rgba(15, 15, 25, 0.9);
    --accent: #646cff;
    --accent-hover: #535bf2;
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.8);
    --border: rgba(255, 255, 255, 0.1);
  }

  body {
    background-color: var(--dark-bg);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    -webkit-font-smoothing: antialiased;
  }
`;

const Layout = styled.div`
  display: flex;
  min-height: 100vh;
`;

const Sidebar = styled.div`
  width: 280px;
  background-color: var(--sidebar-bg);
  padding: 1.5rem;
  border-right: 1px solid var(--border);
  backdrop-filter: blur(16px);
  height: 100vh;
  position: sticky;
  top: 0;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
`;

const LogoText = styled.h1`
  font-size: 1.25rem;
  font-weight: 600;
  margin: 0;
  color: var(--text-primary);
  background: linear-gradient(90deg, #646cff, #61dafb);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
`;

const NavMenu = styled.nav`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 1rem;
`;

const NavLink = styled(Link)`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  color: var(--text-secondary);
  text-decoration: none;
  transition: all 0.2s ease;
  font-size: 0.95rem;

  &:hover {
    background-color: rgba(100, 108, 255, 0.1);
    color: var(--accent);
  }

  &.active {
    background-color: rgba(100, 108, 255, 0.2);
    color: var(--accent);
    font-weight: 500;
  }

  svg {
    width: 20px;
    height: 20px;
  }
`;

const MainContent = styled.main`
  flex: 1;
  padding: 2rem;
  overflow-y: auto;
`;

const UploadCard = styled.div`
  background: var(--card-bg);
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  border: 1px solid var(--border);
  backdrop-filter: blur(10px);
  max-width: 600px;
  margin: 0 auto;
  transition: all 0.3s ease;

  &:hover {
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
  }
`;

const Title = styled.h2`
  font-size: 1.75rem;
  color: var(--text-primary);
  text-align: center;
  font-weight: 600;
  margin-bottom: 2rem;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background: var(--accent);
    border-radius: 3px;
  }
`;

const InputLabel = styled.label`
  display: block;
  margin-bottom: 0.5rem;
  color: var(--text-primary);
  font-size: 0.9rem;
  font-weight: 500;
`;

const Input = styled.input`
  width: 100%;
  margin-bottom: 1.5rem;
  padding: 0.75rem 1rem;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: rgba(10, 10, 15, 0.5);
  color: var(--text-primary);
  font-size: 0.95rem;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
  
  &:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(100, 108, 255, 0.2);
  }
`;

const FileInputWrapper = styled.div`
  position: relative;
  margin-bottom: 1.5rem;
`;

const FileInputLabel = styled.label`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  border: 2px dashed var(--border);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.02);
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    border-color: var(--accent);
    background: rgba(255, 255, 255, 0.05);
  }
`;

const FileInputText = styled.span`
  color: var(--text-primary);
  font-size: 0.95rem;
  margin-top: 0.5rem;
  text-align: center;
`;

const FileName = styled.div`
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: var(--accent);
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
`;

const UploadButton = styled.button`
  width: 100%;
  padding: 0.85rem;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 10px;
  font-weight: 500;
  font-size: 1rem;
  cursor: pointer;
  box-shadow: 0 4px 15px rgba(100, 108, 255, 0.3);
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(100, 108, 255, 0.4);
    background: var(--accent-hover);
  }

  &:disabled {
    opacity: 0.7;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const ResponseBox = styled.div`
  margin-top: 1.5rem;
  padding: 1.25rem;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  color: var(--text-primary);
  font-size: 0.9rem;
  overflow-x: auto;
  max-height: 300px;
  transition: all 0.3s ease;
  
  pre {
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
`;

const StatusMessage = styled.div`
  padding: 0.75rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: ${({ type }) => 
    type === 'error' ? 'rgba(255, 0, 0, 0.1)' : 'rgba(0, 255, 0, 0.1)'};
  color: ${({ type }) => 
    type === 'error' ? '#ff6b6b' : '#51cf66'};
  border-left: 3px solid ${({ type }) => 
    type === 'error' ? '#ff6b6b' : '#51cf66'};
`;

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [groundStation, setGroundStation] = useState('');
  const [response, setResponse] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);

  const handleFileChange = useCallback((e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setStatus(null);
    }
  }, []);

  const handleUpload = async () => {
    if (!file) {
      setStatus({ type: 'error', message: 'Please select a file.' });
      return;
    }
    if (!groundStation.trim()) {
      setStatus({ type: 'error', message: 'Please enter a ground station name.' });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('ground_station', groundStation.trim());

    let endpoint = '';
    const filename = file.name.toLowerCase();

    if (filename.endsWith('.zip')) {
      endpoint = 'http://127.0.0.1:8000/process/zip';
    } else if (filename.endsWith('.xlsx')) {
      endpoint = 'http://127.0.0.1:8000/process/excel';
    } else {
      setStatus({ type: 'error', message: 'Unsupported file type. Please upload a .zip or .xlsx file.' });
      return;
    }

    try {
      setUploading(true);
      setStatus({ type: 'info', message: 'Uploading file...' });
      
      const res = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      setResponse(res.data);
      setStatus({ type: 'success', message: 'File processed successfully!' });
    } catch (error) {
      console.error(error);
      const errorMessage = error.response?.data?.message || 'Upload failed. Please try again.';
      setResponse({ error: errorMessage });
      setStatus({ type: 'error', message: errorMessage });
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      <GlobalStyle />
      <Layout>
        <Sidebar>
                  <Logo>
                    <img src={logo1} alt="Logo" width={32} height={32} />
                    <LogoText>Navavishkar</LogoText>
                  </Logo>
                  <NavMenu>
                    <NavLink to="/home">Home</NavLink>
                    <NavLink to="/dashboard">Dashboard</NavLink>
                    <NavLink to="/upload">Upload</NavLink> {/* ✅ NEW LINK */}
                    <NavLink to="/database">Database</NavLink>
                    <NavLink to="/settings">Settings</NavLink>
                  </NavMenu>
                </Sidebar>
        

        <MainContent>
          <UploadCard>
            <Title>Upload Raw Data</Title>
            
            <div>
              <InputLabel htmlFor="groundStation">Ground Station Name</InputLabel>
              <Input
                id="groundStation"
                type="text"
                placeholder="e.g. GS-1, Main Station, etc."
                value={groundStation}
                onChange={(e) => setGroundStation(e.target.value)}
                disabled={uploading}
              />
            </div>
            
            <FileInputWrapper>
              <InputLabel>Data File</InputLabel>
              <FileInputLabel>
                <input 
                  type="file" 
                  onChange={handleFileChange} 
                  style={{ display: 'none' }}
                  accept=".zip,.xlsx"
                  disabled={uploading}
                />
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                <FileInputText>
                  {file ? 'Click to change file' : 'Click to browse or drag and drop'}
                </FileInputText>
                {file && <FileName>{file.name}</FileName>}
              </FileInputLabel>
            </FileInputWrapper>
            
            {status && (
              <StatusMessage type={status.type}>
                {status.type === 'error' ? (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M8 12l3 3 5-6"></path>
                  </svg>
                )}
                {status.message}
              </StatusMessage>
            )}
            
            <UploadButton onClick={handleUpload} disabled={uploading}>
              {uploading ? (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                  Uploading...
                </>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                  Upload File
                </>
              )}
            </UploadButton>

            {response && (
              <ResponseBox>
                <pre>{JSON.stringify(response, null, 2)}</pre>
              </ResponseBox>
            )}
          </UploadCard>
        </MainContent>
      </Layout>
    </>
  );
};

export default FileUpload;