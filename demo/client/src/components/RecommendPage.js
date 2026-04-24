import { useCallback, useState } from 'react';
import { Row, Col, Button, Label, Input } from 'reactstrap';
import ResultRow from './ResultRow';
import useLocalStorage from './hooks/useLocalStorage';

function RecommendPage() {
  const [cards, setCards] = useLocalStorage('recommendinput', '');
  const [adds, setAdds] = useState([]);
  const [removes, setRemoves] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchCubes = useCallback(async () => {
    setLoading(true);
    try {
      const cubeNames = cards.split('\n').map(s => s.trim()).filter(Boolean);

      const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cards: cubeNames }),
      });
      const json = await response.json();

      setAdds(json.adds);
      setRemoves(json.removes);
    } finally {
      setLoading(false);
    }
  }, [cards]);

  return (
    <Row className="h-100 g-3">
      <Col md="4" className="h-100 overflow-auto">
        <Label className="fw-bold mb-1">Cube</Label>
        <Input type="textarea" rows="18" value={cards} onChange={e => setCards(e.target.value)} />
        <Button block color="primary" className="mt-3 w-100" onClick={fetchCubes} disabled={loading}>
          {loading ? 'Fetching…' : 'Fetch'}
        </Button>
      </Col>
      <Col md="8" className="h-100 overflow-auto">
        <Row className="g-3">
          <Col md="6">
            <h5>Recommended Adds</h5>
            {adds.length === 0 && <div className="text-muted small">No results yet.</div>}
            {adds.map((card, i) => (
              <ResultRow key={`add-${card.name}-${i}`} index={i} card={card} />
            ))}
          </Col>
          <Col md="6">
            <h5>Recommended Removes</h5>
            {removes.length === 0 && <div className="text-muted small">No results yet.</div>}
            {removes.map((card, i) => (
              <ResultRow key={`rem-${card.name}-${i}`} index={i} card={card} ratingFormat="inverse" />
            ))}
          </Col>
        </Row>
      </Col>
    </Row>
  );
}

export default RecommendPage;
