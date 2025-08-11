// UPDATED Dashboard.jsx (first graph connected to API with date range and model selection, original/predicted differentiated)

import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import flatpickr from "flatpickr";
import "flatpickr/dist/flatpickr.min.css";
import Plotly from "plotly.js-dist";
import styled, { createGlobalStyle } from "styled-components";
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
  flex-direction: column;

  @media (min-width: 1024px) {
    flex-direction: row;
  }
`;

const Sidebar = styled.div`
  width: 100%;
  background-color: var(--sidebar-bg);
  padding: 1rem;
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(16px);
  position: sticky;
  top: 0;
  z-index: 100;
  display: flex;
  flex-direction: column;

  @media (min-width: 1024px) {
    width: 280px;
    height: 100vh;
    border-right: 1px solid var(--border);
    border-bottom: none;
    padding: 1.5rem;
  }
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);

  @media (min-width: 1024px) {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
  }
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
  padding: 1.5rem;
  overflow-y: auto;

  @media (min-width: 768px) {
    padding: 2rem;
  }
`;

const GraphContainer = styled.div`
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

const GraphHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 1rem;
`;

const GraphTitle = styled.h2`
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
`;

const DateInputGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
`;

const DateInput = styled.input`
  padding: 0.75rem 1rem;
  background: rgba(10, 10, 15, 0.5);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text-primary);
  font-size: 0.9rem;
  min-width: 150px;
  transition: all 0.3s ease;

  &:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(100, 108, 255, 0.2);
  }
`;

const Separator = styled.span`
  color: var(--text-secondary);
`;

const Dropdown = styled.select`
  padding: 0.6rem 1rem;
  border-radius: 8px;
  background-color: rgba(10, 10, 15, 0.5);
  color: var(--text-primary);
  border: 1px solid var(--border);
  font-size: 0.9rem;

  &:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(100, 108, 255, 0.2);
  }
`;

// Styled components assumed unchanged from your working version
// Layout, Sidebar, Logo, LogoText, NavMenu, NavLink, MainContent, GraphContainer, GraphHeader, GraphTitle, DateInputGroup, DateInput, Separator, Dropdown

const Dashboard = () => {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [selectedDate, setSelectedDate] = useState("");
  const [modelType, setModelType] = useState("1");

  useEffect(() => {
    flatpickr("#startDate", {
      dateFormat: "Y-m-d",
      defaultDate: startDate,
      onChange: (dates) => setStartDate(dates[0]?.toISOString().split('T')[0] || "")
    });

    flatpickr("#endDate", {
      dateFormat: "Y-m-d",
      defaultDate: endDate,
      onChange: (dates) => setEndDate(dates[0]?.toISOString().split('T')[0] || "")
    });

    flatpickr("#selectedDate", {
      dateFormat: "Y-m-d",
      defaultDate: selectedDate,
      onChange: (dates) => setSelectedDate(dates[0]?.toISOString().split('T')[0] || "")
    });
  }, []);

  useEffect(() => {
    if (!startDate || !endDate || !modelType) return;

const fetchData = async () => {
  try {
    const response = await fetch(`http://localhost:5000/api/model-predict-range?model=${modelType}&start=${startDate}&end=${endDate}`);
    const data = await response.json();

    const groupedByDay = {};

    if (Array.isArray(data.data)) {
      for (const row of data.data) {
        const { time, tec, source } = row;
        const day = time.split(" ")[0];

        if (!groupedByDay[day]) {
          groupedByDay[day] = { predicted: [], original: [] };
        }

        groupedByDay[day][source].push({ x: time, y: tec });
      }
    }

    const graphData = [];

    for (const [day, { predicted, original }] of Object.entries(groupedByDay)) {
      if (predicted.length) {
        graphData.push({
          x: predicted.map(d => d.x),
          y: predicted.map(d => d.y),
          type: 'scatter',
          mode: 'lines+markers',
          name: `Predicted ${day}`,
          line: { color: '#ff4d4d', shape: 'spline', width: 3 },
          marker: { color: '#ff4d4d' }
        });
      }

      if (original.length) {
        // add the last predicted point as the first of original ONLY IF time gap is short
        if (predicted.length) {
          const lastPredTime = new Date(predicted[predicted.length - 1].x);
          const firstOrigTime = new Date(original[0].x);
          const diff = (firstOrigTime - lastPredTime) / 1000; // seconds

          if (diff <= 3600) {  // if within 1 hour, connect
            original.unshift({
              x: predicted[predicted.length - 1].x,
              y: predicted[predicted.length - 1].y
            });
          }
        }

        graphData.push({
          x: original.map(d => d.x),
          y: original.map(d => d.y),
          type: 'scatter',
          mode: 'lines+markers',
          name: `Original ${day}`,
          line: { color: '#00ccff', shape: 'spline', width: 3 },
          marker: { color: '#00ccff' }
        });
      }
    }

    Plotly.newPlot("top-graph", graphData, {
      plot_bgcolor: "rgba(0,0,0,0)",
      paper_bgcolor: "rgba(0,0,0,0)",
      font: { color: '#fff', family: 'Inter' },
      xaxis: { title: "Time", showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
      yaxis: { title: "TEC", showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
      title: `Prediction: Model ${modelType}`,
      margin: { t: 40, r: 40, b: 60, l: 60 },
      hovermode: 'closest',
      hoverlabel: { bgcolor: 'rgba(30,30,40,0.9)', bordercolor: 'rgba(100,108,255,0.5)' }
    });
  } catch (error) {
    console.error("Error fetching prediction data:", error);
    Plotly.purge("top-graph");
  }
};


    fetchData();
  }, [startDate, endDate, modelType]);
useEffect(() => {
  if (!selectedDate || !modelType) return;

  const fetchMidGraph = async () => {
    try {
      const response = await fetch(`http://localhost:5000/api/model-predict-date?model=${modelType}&date=${selectedDate}`);
      const data = await response.json();

      const predicted = [];
      const original = [];

      if (Array.isArray(data.data)) {
        for (const row of data.data) {
          const { time, tec, source } = row;
          if (source === "predicted") {
            predicted.push({ x: time, y: tec });
          } else {
            original.push({ x: time, y: tec });
          }
        }
      }

      const graphData = [];

      if (predicted.length) {
        graphData.push({
          x: predicted.map(d => d.x),
          y: predicted.map(d => d.y),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Predicted',
          line: { color: '#ff4d4d', shape: 'spline', width: 3 },
          marker: { color: '#ff4d4d' }
        });
      }

      if (original.length) {
        // Connect last predicted to first original if close in time
        if (predicted.length) {
          const lastPredTime = new Date(predicted[predicted.length - 1].x);
          const firstOrigTime = new Date(original[0].x);
          const diff = (firstOrigTime - lastPredTime) / 1000;

          if (diff <= 3600) {
            const merged = [...predicted.slice(-1), ...original];
            merged.sort((a, b) => new Date(a.x) - new Date(b.x));

            graphData.push({
              x: merged.map(d => d.x),
              y: merged.map(d => d.y),
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Original',
              line: { color: '#00ccff', shape: 'spline', width: 3 },
              marker: { color: '#00ccff' }
            });
          } else {
            graphData.push({
              x: original.map(d => d.x),
              y: original.map(d => d.y),
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Original',
              line: { color: '#00ccff', shape: 'spline', width: 3 },
              marker: { color: '#00ccff' }
            });
          }
        }
      }

      Plotly.newPlot("mid-graph", graphData, {
        plot_bgcolor: "rgba(0,0,0,0)",
        paper_bgcolor: "rgba(0,0,0,0)",
        font: { color: '#fff', family: 'Inter' },
        xaxis: { title: "Time", showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { title: "TEC", showgrid: true, gridcolor: 'rgba(255,255,255,0.1)' },
        title: `Detailed Analysis: ${selectedDate}`,
        margin: { t: 40, r: 40, b: 60, l: 60 },
        hovermode: 'closest',
        hoverlabel: { bgcolor: 'rgba(30,30,40,0.9)', bordercolor: 'rgba(100,108,255,0.5)' }
      });

    } catch (error) {
      console.error("Error fetching detailed data:", error);
      Plotly.purge("mid-graph");
    }
  };

  fetchMidGraph();
}, [selectedDate, modelType]);

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
          <GraphContainer>
            <GraphHeader>
              <GraphTitle>Daily GNSS Data</GraphTitle>
              <DateInputGroup>
                <DateInput type="text" id="startDate" placeholder="Start Date" />
                <Separator>to</Separator>
                <DateInput type="text" id="endDate" placeholder="End Date" />
                <Dropdown value={modelType} onChange={(e) => setModelType(e.target.value)}>
                  <option value="1">Random Forest</option>
                  <option value="2">Gradient Boosting</option>
                  <option value="3">XGBoost</option>
                  <option value="4">MLP</option>
                  <option value="5">LSTM</option>
                  <option value="6">BiLSTM</option>
                </Dropdown>
              </DateInputGroup>
            </GraphHeader>
            <div id="top-graph" style={{ height: "610px" }} />
          </GraphContainer>
        </MainContent>
      </Layout>
    </>
  );
};

export default Dashboard;
