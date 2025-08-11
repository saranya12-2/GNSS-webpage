import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import styled, { createGlobalStyle } from 'styled-components';
import axios from 'axios';
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
    --success: #51cf66;
    --warning: #fcc419;
    --danger: #ff6b6b;
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

const Card = styled.div`
  background: var(--card-bg);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  border: 1px solid var(--border);
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;

  &:hover {
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
  }
`;

const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 1rem;
`;

const CardTitle = styled.h2`
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
`;

const RefreshButton = styled.button`
  padding: 0.5rem 1rem;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  font-size: 0.9rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.3s ease;

  &:hover {
    background: var(--accent-hover);
    transform: translateY(-2px);
  }
`;

const TableContainer = styled.div`
  overflow-x: auto;
  margin-top: 1rem;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  color: var(--text-primary);
`;

const TableHead = styled.thead`
  background: rgba(100, 108, 255, 0.1);
`;

const TableRow = styled.tr`
  &:nth-child(even) {
    background: rgba(255, 255, 255, 0.05);
  }

  &:hover {
    background: rgba(100, 108, 255, 0.15);
  }
`;

const TableHeader = styled.th`
  padding: 1rem;
  text-align: left;
  font-weight: 600;
  border-bottom: 1px solid var(--border);
`;

const TableCell = styled.td`
  padding: 1rem;
  border-bottom: 1px solid var(--border);
`;

const StatusBadge = styled.span`
  display: inline-block;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 500;
  background: ${({ status }) => 
    status === 'active' ? 'rgba(81, 207, 102, 0.1)' : 
    status === 'inactive' ? 'rgba(255, 107, 107, 0.1)' : 
    'rgba(252, 196, 25, 0.1)'};
  color: ${({ status }) => 
    status === 'active' ? 'var(--success)' : 
    status === 'inactive' ? 'var(--danger)' : 
    'var(--warning)'};
`;

const StatsContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const StatCard = styled.div`
  background: var(--card-bg);
  border-radius: 12px;
  padding: 1.5rem;
  border: 1px solid var(--border);
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
  }
`;

const StatTitle = styled.h3`
  margin: 0 0 0.5rem 0;
  font-size: 1rem;
  font-weight: 500;
  color: var(--text-secondary);
`;

const StatValue = styled.p`
  margin: 0;
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--text-primary);
`;

const DatabaseMonitor = () => {
  const [databaseStats, setDatabaseStats] = useState(null);
  const [tables, setTables] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchDatabaseInfo = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Simulate API calls - replace with your actual endpoints
      const statsResponse = await axios.get('http://localhost:5000/api/database/stats');
      const tablesResponse = await axios.get('http://localhost:5000/api/database/tables');
      
      setDatabaseStats(statsResponse.data);
      setTables(tablesResponse.data);
    } catch (err) {
      console.error('Error fetching database info:', err);
      setError('Failed to load database information');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatabaseInfo();
  }, []);

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
          <h1 style={{ marginTop: 0 }}>Database Monitoring</h1>

          {loading && <p>Loading database information...</p>}
          {error && <p style={{ color: 'var(--danger)' }}>{error}</p>}

          {databaseStats && (
            <>
              <StatsContainer>
                <StatCard>
                  <StatTitle>Database Size</StatTitle>
                  <StatValue>{databaseStats.size}</StatValue>
                </StatCard>
                <StatCard>
                  <StatTitle>Total Tables</StatTitle>
                  <StatValue>{databaseStats.tableCount}</StatValue>
                </StatCard>
                <StatCard>
                  <StatTitle>Total Records</StatTitle>
                  <StatValue>{databaseStats.recordCount.toLocaleString()}</StatValue>
                </StatCard>
                <StatCard>
                  <StatTitle>Uptime</StatTitle>
                  <StatValue>{databaseStats.uptime}</StatValue>
                </StatCard>
              </StatsContainer>

              <Card>
                <CardHeader>
                  <CardTitle>Database Tables</CardTitle>
                  <RefreshButton onClick={fetchDatabaseInfo}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M23 4v6h-6"></path>
                      <path d="M1 20v-6h6"></path>
                      <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
                    </svg>
                    Refresh
                  </RefreshButton>
                </CardHeader>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableHeader>Table Name</TableHeader>
                        <TableHeader>Records</TableHeader>
                        <TableHeader>Size</TableHeader>
                        <TableHeader>Last Updated</TableHeader>
                        <TableHeader>Status</TableHeader>
                      </TableRow>
                    </TableHead>
                    <tbody>
                      {tables.map((table) => (
                        <TableRow key={table.name}>
                          <TableCell>{table.name}</TableCell>
                          <TableCell>{table.records.toLocaleString()}</TableCell>
                          <TableCell>{table.size}</TableCell>
                          <TableCell>{table.lastUpdated}</TableCell>
                          <TableCell>
                            <StatusBadge status={table.status}>
                              {table.status}
                            </StatusBadge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </tbody>
                  </Table>
                </TableContainer>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Query Performance</CardTitle>
                </CardHeader>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableHeader>Query</TableHeader>
                        <TableHeader>Avg Execution Time</TableHeader>
                        <TableHeader>Calls</TableHeader>
                        <TableHeader>Last Executed</TableHeader>
                      </TableRow>
                    </TableHead>
                    <tbody>
                      {databaseStats.queryStats.map((query, index) => (
                        <TableRow key={index}>
                          <TableCell style={{ maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {query.query}
                          </TableCell>
                          <TableCell>{query.avgTime} ms</TableCell>
                          <TableCell>{query.calls}</TableCell>
                          <TableCell>{query.lastExecuted}</TableCell>
                        </TableRow>
                      ))}
                    </tbody>
                  </Table>
                </TableContainer>
              </Card>
            </>
          )}
        </MainContent>
      </Layout>
    </>
  );
};

export default DatabaseMonitor;