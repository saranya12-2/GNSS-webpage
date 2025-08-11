import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import styled, { createGlobalStyle, ThemeProvider } from 'styled-components';
import Particles from './Particles';
import logo1 from '../assets/logo1.png';
import logo2 from '../assets/logo2.png';

// Professional Dark Theme
const darkTheme = {
  body: '#0a0a0f',
  navBg: 'rgba(14, 12, 30, 0.9)',
  text: '#ffffff',
  subtitle: 'rgba(255, 255, 255, 0.85)',
  accent: '#646cff',
  accentLight: '#848bff',
  particleColors: ["#646cff", "#535bf2", "#61dafb"],
  buttonBg: 'linear-gradient(135deg, #646cff, #535bf2)',
  glassBorder: 'rgba(255, 255, 255, 0.1)',
  cardBg: 'rgba(25, 25, 40, 0.7)'
};

const lightTheme = {
  body: '#f8f9fa',
  navBg: 'rgba(248, 249, 250, 0.9)',
  text: '#1a1a1a',
  subtitle: 'rgba(26, 26, 26, 0.85)',
  accent: '#646cff',
  accentLight: '#848bff',
  particleColors: ["#646cff", "#535bf2", "#61dafb"],
  buttonBg: 'linear-gradient(135deg, #646cff, #535bf2)',
  glassBorder: 'rgba(0, 0, 0, 0.1)',
  cardBg: 'rgba(255, 255, 255, 0.7)'
};

const GlobalStyle = createGlobalStyle`
  html, body, #root {
    margin: 0;
    padding: 0;
    height: 100%;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: ${({ theme }) => theme.body};
    color: ${({ theme }) => theme.text};
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
  }

  * {
    box-sizing: border-box;
  }
`;

const HomeContainer = styled.div`
  position: relative;
  width: 100vw;
  min-height: 100vh;
  overflow-x: hidden;
`;

const Navbar = styled.nav`
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: ${({ theme }) => theme.navBg};
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  position: sticky;
  top: 0;
  z-index: 100;
  border-bottom: 1px solid ${({ theme }) => theme.glassBorder};

  @media (max-width: 768px) {
    padding: 1rem;
    flex-direction: column;
    gap: 1rem;
  }
`;

const LogoWrapper = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;

  @media (max-width: 768px) {
    width: 100%;
    justify-content: center;
  }
`;

const Logo = styled.img`
  height: 50px;
  width: auto;

  @media (max-width: 480px) {
    height: 35px;
  }
`;

const InstitutionName = styled(Link)`
  color: ${({ theme }) => theme.text};
  text-decoration: none;
  font-size: 1.3rem;
  font-weight: 600;
  letter-spacing: -0.5px;
  transition: color 0.3s ease;
  
  &:hover {
    color: ${({ theme }) => theme.accent};
  }

  @media (max-width: 768px) {
    font-size: 1.1rem;
  }

  @media (max-width: 480px) {
    font-size: 1rem;
  }
`;

const NavLinks = styled.div`
  display: flex;
  gap: 1.5rem;
  align-items: center;

  @media (max-width: 768px) {
    width: 100%;
    justify-content: center;
    gap: 1rem;
  }

  @media (max-width: 480px) {
    flex-wrap: wrap;
  }
`;

const NavLink = styled(Link)`
  color: ${({ theme }) => theme.text};
  text-decoration: none;
  font-weight: 500;
  font-size: 1rem;
  transition: all 0.3s ease;
  padding: 0.5rem 0;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 0;
    height: 2px;
    background: ${({ theme }) => theme.accent};
    transition: width 0.3s ease;
  }
  
  &:hover::after {
    width: 100%;
  }

  @media (max-width: 480px) {
    font-size: 0.9rem;
    padding: 0.3rem 0;
  }
`;

const ThemeToggle = styled.button`
  padding: 0.5rem 1rem;
  border-radius: 6px;
  border: none;
  background: rgba(100, 108, 255, 0.1);
  color: ${({ theme }) => theme.text};
  font-weight: 500;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover {
    background: rgba(100, 108, 255, 0.2);
  }

  @media (max-width: 480px) {
    padding: 0.4rem 0.8rem;
    font-size: 0.8rem;
  }
`;

const MainContent = styled.main`
  position: relative;
  z-index: 10;
  height: calc(100vh - 80px);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 0 1.5rem;

  @media (max-width: 768px) {
    height: auto;
    padding: 2rem 1rem;
  }
`;

const GlassCard = styled.div`
  // background: ${({ theme }) => theme.cardBg};
  // backdrop-filter: blur(2px);
  // -webkit-backdrop-filter: blur(2px);
  // border-radius: 20px;
  // border: 1px solid ${({ theme }) => theme.glassBorder};
  // padding: 3rem;
  // max-width: 800px;
  // width: 90%;
  // box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  // margin: 2rem 0;

  // @media (max-width: 768px) {
  //   padding: 2rem;
  // }

  // @media (max-width: 480px) {
  //   padding: 1.5rem;
  //   border-radius: 16px;
  // }
`;

const Title = styled.h1`
  font-size: 2.5rem;
  margin-bottom: 1.5rem;
  font-weight: 700;
  line-height: 1.3;
  color: ${({ theme }) => theme.text};
  letter-spacing: -0.5px;

  @media (max-width: 768px) {
    font-size: 2rem;
  }

  @media (max-width: 480px) {
    font-size: 1.8rem;
    margin-bottom: 1rem;
  }
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  color: ${({ theme }) => theme.subtitle};
  margin-bottom: 2.5rem;
  max-width: 700px;
  line-height: 1.6;
  font-weight: 400;

  @media (max-width: 768px) {
    font-size: 1.1rem;
    margin-bottom: 2rem;
  }

  @media (max-width: 480px) {
    font-size: 1rem;
    line-height: 1.5;
  }
`;

const DashboardButton = styled(Link)`
  padding: 0.9rem 2rem;
  background: ${({ theme }) => theme.buttonBg};
  color: white;
  border-radius: 8px;
  text-decoration: none;
  font-weight: 500;
  font-size: 1rem;
  transition: all 0.3s ease;
  border: none;
  box-shadow: 0 4px 20px rgba(100, 108, 255, 0.3);
  
  &:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(100, 108, 255, 0.4);
  }

  @media (max-width: 480px) {
    padding: 0.8rem 1.5rem;
    font-size: 0.9rem;
  }
`;

const ParticleContainer = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
`;

const HomePage = () => {
  const [isDark, setIsDark] = useState(true);
  const [particleCount, setParticleCount] = useState(700);

  useEffect(() => {
    const handleResize = () => {
      // Adjust particle count based on screen size
      if (window.innerWidth < 768) {
        setParticleCount(400);
      } else if (window.innerWidth < 1024) {
        setParticleCount(550);
      } else {
        setParticleCount(700);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const toggleTheme = () => {
    setIsDark((prev) => !prev);
  };

  return (
    <ThemeProvider theme={isDark ? darkTheme : lightTheme}>
      <>
        <GlobalStyle />
        <ParticleContainer>
          <Particles
            particleCount={particleCount}
            particleSpread={12}
            speed={0.15}
            particleColors={isDark ? darkTheme.particleColors : lightTheme.particleColors}
            moveParticlesOnHover={true}
            particleHoverFactor={6}
            alphaParticles={true}
            particleBaseSize={window.innerWidth < 768 ? 50 : 60}
            sizeRandomness={0.7}
            cameraDistance={15}
            disableRotation={false}
          />
        </ParticleContainer>
        
        <HomeContainer>
          <Navbar>
            <LogoWrapper>
              <Logo src={logo1} alt="IIT Tirupati Logo" />
              <Logo src={logo2} alt="Navavishkar Logo" />
        
            </LogoWrapper>
            <NavLinks>
              <NavLink to="/home">Home</NavLink>
              <NavLink to="/dashboard">Dashboard</NavLink>
              <NavLink to="/upload">Upload</NavLink> {/* ✅ NEW LINK */}
              <NavLink to="/database">Database</NavLink>
              <NavLink to="/about">About</NavLink>
              <ThemeToggle onClick={toggleTheme}>
                {isDark ? 'Light' : 'Dark'}
              </ThemeToggle>
            </NavLinks>
          </Navbar>


          <MainContent>
            <GlassCard>
              <Title>IIT Tirupati Navavishkar I-Hub Foundation</Title>
              <Subtitle>
                Advanced visualization and analytical tools for Global Navigation Satellite System research.
                Precise positioning, timing, and atmospheric studies with real-time data processing.
              </Subtitle>
            
            </GlassCard>
          </MainContent>
        </HomeContainer>
      </>
    </ThemeProvider>
  );
};

export default HomePage;