import 'bootstrap/dist/css/bootstrap.min.css';

import { useState } from 'react';
import { Card, CardBody, Container, Nav, NavItem, NavLink } from 'reactstrap'

import BuildPage from './components/BuildPage';
import DraftPage from './components/DraftPage';
import RecommendPage from './components/RecommendPage';

function App() {
  const [tab, setTab] = useState(0);


  return (
    <Card className="m-4">
      <Nav tabs className="mt-2">
        <NavItem>
          <NavLink href="#" active={tab === 0} onClick={() => setTab(0)}>Recommend</NavLink>
        </NavItem>
        <NavItem>
          <NavLink href="#" active={tab === 1} onClick={() => setTab(1)}>Deck Build</NavLink>
        </NavItem>
        <NavItem>
          <NavLink href="#" active={tab === 2} onClick={() => setTab(2)}>Draft</NavLink>
        </NavItem>
      </Nav>
      <CardBody>
        {tab === 0 && <RecommendPage />}
        {tab === 1 && <BuildPage />}
        {tab === 2 && <DraftPage />}
      </CardBody>
    </Card>
  );
}

export default App;
