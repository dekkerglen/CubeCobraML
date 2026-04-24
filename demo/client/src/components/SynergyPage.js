import { useCallback, useState } from 'react';
import { Row, Col, Button, Label, Input } from 'reactstrap';
import ResultRow from './ResultRow';
import useLocalStorage from './hooks/useLocalStorage';

function SynergyPage() {
  const [card, setCard] = useLocalStorage('synergy', '');
  const [synergies, setSynergies] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchSynergies = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/synergies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ card }),
      });
      const json = await response.json();
      setSynergies(json.cards);
    } finally {
      setLoading(false);
    }
  }, [card]);

  return (
    <Row className="h-100 g-3">
      <Col md="4" className="h-100 overflow-auto">
        <Label className="fw-bold mb-1">Card Name</Label>
        <Input type="text" value={card} onChange={e => setCard(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && fetchSynergies()} />
        <Button block color="primary" className="mt-3 w-100" onClick={fetchSynergies} disabled={loading}>
          {loading ? 'Fetching…' : 'Fetch'}
        </Button>
      </Col>
      <Col md="8" className="h-100 overflow-auto">
        <h5>Synergy With</h5>
        {synergies.length === 0 && <div className="text-muted small">No results yet.</div>}
        {synergies.map((c, i) => (
          <ResultRow key={`${c.name}-${i}`} index={i} card={c} />
        ))}
      </Col>
    </Row>
  );
}

export default SynergyPage;
