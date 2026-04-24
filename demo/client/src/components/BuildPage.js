import { useCallback, useState } from 'react';
import { Row, Col, Button, Label, Input } from 'reactstrap';
import ResultRow from './ResultRow';
import useLocalStorage from './hooks/useLocalStorage';

function BuildPage() {
  const [cards, setCards] = useLocalStorage('buildinput', '');
  const [cubeText, setCubeText] = useLocalStorage('buildcubeinput', '');
  const [mainboard, setMainboard] = useState([]);
  const [sideboard, setSideboard] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchPools = useCallback(async () => {
    setLoading(true);
    try {
      const poolNames = cards.split('\n').map(s => s.trim()).filter(Boolean);
      const cubeNames = cubeText.split('\n').map(s => s.trim()).filter(Boolean);

      const response = await fetch('/api/deckbuild', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cards: poolNames, cube: cubeNames }),
      });
      const json = await response.json();

      setMainboard(json.mainboard);
      setSideboard(json.sideboard);
    } finally {
      setLoading(false);
    }
  }, [cards, cubeText]);

  return (
    <Row className="h-100 g-3">
      <Col md="4" className="h-100 overflow-auto">
        <Label className="fw-bold mb-1">Pool</Label>
        <Input type="textarea" rows="10" value={cards} onChange={e => setCards(e.target.value)} />

        <Label className="fw-bold mt-3 mb-1">Cube Context</Label>
        <Input type="textarea" rows="8" value={cubeText} onChange={e => setCubeText(e.target.value)} />

        <Button block color="primary" className="mt-3 w-100" onClick={fetchPools} disabled={loading}>
          {loading ? 'Fetching…' : 'Fetch'}
        </Button>
      </Col>
      <Col md="8" className="h-100 overflow-auto">
        <Row className="g-3">
          <Col md="6">
            <h5>Mainboard</h5>
            {mainboard.length === 0 && <div className="text-muted small">No results yet.</div>}
            {mainboard.map((card, i) => (
              <ResultRow key={`main-${card.name}-${i}`} index={i} card={card} />
            ))}
          </Col>
          <Col md="6">
            <h5>Sideboard</h5>
            {sideboard.length === 0 && <div className="text-muted small">No results yet.</div>}
            {sideboard.map((card, i) => (
              <ResultRow key={`side-${card.name}-${i}`} index={i} card={card} ratingFormat="inverse" />
            ))}
          </Col>
        </Row>
      </Col>
    </Row>
  );
}

export default BuildPage;
