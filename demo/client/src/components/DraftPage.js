import { useCallback, useState } from 'react';
import { Row, Col, Button, Label, Input } from 'reactstrap';
import ResultRow from './ResultRow';
import useLocalStorage from './hooks/useLocalStorage';

function DraftPage() {
  const [packText, setPackText] = useLocalStorage('packinput', '');
  const [poolText, setPoolText] = useLocalStorage('poolinput', '');
  const [cubeText, setCubeText] = useLocalStorage('draftcubeinput', '');
  const [landPctText, setLandPctText] = useLocalStorage('draftlandpct', '0');
  const [nonlandPctText, setNonlandPctText] = useLocalStorage('draftnonlandpct', '0');
  const [picks, setPicks] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchPicks = useCallback(async () => {
    setLoading(true);
    try {
      const packNames = packText.split('\n').map(s => s.trim()).filter(Boolean);
      const poolNames = poolText.split('\n').map(s => s.trim()).filter(Boolean);
      const cubeNames = cubeText.split('\n').map(s => s.trim()).filter(Boolean);
      const landPct = parseFloat(landPctText) || 0;
      const nonlandPct = parseFloat(nonlandPctText) || 0;

      const response = await fetch('/api/draft', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pack: packNames, pool: poolNames, cube: cubeNames, landPct, nonlandPct }),
      });
      const json = await response.json();
      setPicks(json.picks);
    } finally {
      setLoading(false);
    }
  }, [packText, poolText, cubeText, landPctText, nonlandPctText]);

  return (
    <Row className="h-100 g-3">
      <Col md="4" className="h-100 overflow-auto">
        <Label className="fw-bold mb-1">Pack</Label>
        <Input type="textarea" rows="5" value={packText} onChange={e => setPackText(e.target.value)} />

        <Label className="fw-bold mt-3 mb-1">Pool</Label>
        <Input type="textarea" rows="6" value={poolText} onChange={e => setPoolText(e.target.value)} />

        <Label className="fw-bold mt-3 mb-1">Cube Context</Label>
        <Input type="textarea" rows="6" value={cubeText} onChange={e => setCubeText(e.target.value)} />

        <Row className="mt-3 g-2">
          <Col xs="6">
            <Label size="sm" className="mb-1">Land %</Label>
            <Input type="number" step="0.01" min="0" max="1" bsSize="sm"
              value={landPctText} onChange={e => setLandPctText(e.target.value)} />
          </Col>
          <Col xs="6">
            <Label size="sm" className="mb-1">Nonland %</Label>
            <Input type="number" step="0.01" min="0" max="1" bsSize="sm"
              value={nonlandPctText} onChange={e => setNonlandPctText(e.target.value)} />
          </Col>
        </Row>

        <Button block color="primary" className="mt-3 w-100" onClick={fetchPicks} disabled={loading}>
          {loading ? 'Fetching…' : 'Fetch'}
        </Button>
      </Col>
      <Col md="8" className="h-100 overflow-auto">
        <h5>Recommended Picks</h5>
        {picks.length === 0 && <div className="text-muted small">No picks yet.</div>}
        {picks.map((card, i) => (
          <ResultRow key={`${card.name}-${i}`} index={i} card={card} />
        ))}
      </Col>
    </Row>
  );
}

export default DraftPage;
