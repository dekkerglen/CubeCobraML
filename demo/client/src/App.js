import 'bootstrap/dist/css/bootstrap.min.css';

import { useState } from 'react';
import { Nav, NavItem, NavLink } from 'reactstrap'

import BuildPage from './components/BuildPage';
import DraftPage from './components/DraftPage';
import RecommendPage from './components/RecommendPage';
import SynergyPage from './components/SynergyPage';
import RotoDraftPage from './components/RotoDraftPage';

function App() {
  const [tab, setTab] = useState(0);

  return (
    <div className="d-flex flex-column" style={{ height: '100vh' }}>
      <Nav tabs className="px-3 pt-2 flex-shrink-0">
        <NavItem>
          <NavLink href="#" active={tab === 0} onClick={() => setTab(0)}>Recommend</NavLink>
        </NavItem>
        <NavItem>
          <NavLink href="#" active={tab === 1} onClick={() => setTab(1)}>Deck Build</NavLink>
        </NavItem>
        <NavItem>
          <NavLink href="#" active={tab === 2} onClick={() => setTab(2)}>Draft</NavLink>
        </NavItem>
        <NavItem>
          <NavLink href="#" active={tab === 3} onClick={() => setTab(3)}>Roto Draft</NavLink>
        </NavItem>
        <NavItem>
          <NavLink href="#" active={tab === 4} onClick={() => setTab(4)}>Synergy</NavLink>
        </NavItem>
      </Nav>
      <div className="flex-grow-1 overflow-hidden p-3">
        {tab === 0 && <RecommendPage />}
        {tab === 1 && <BuildPage />}
        {tab === 2 && <DraftPage />}
        {tab === 3 && <RotoDraftPage />}
        {tab === 4 && <SynergyPage />}
      </div>
    </div>
  );
}

export default App;
