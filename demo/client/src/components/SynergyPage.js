import { useCallback, useState } from 'react';
import { Row, Col, Button, Card, CardBody } from 'reactstrap';
import useLocalStorage from './hooks/useLocalStorage';

function SynergyPage() {
  const [card, setCards] = useLocalStorage('synergy', '');
  const [synergies, setSynergies] = useState([]);

  const fetchSynergies = useCallback(async () => {
    const response = await fetch('/api/synergies', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ card }),
    });
    const json = await response.json();    

    setSynergies(json.cards);
  }, [card]);

  return (
    <>
      <h3>Recommend</h3>
      <Row>
        <Col xs="6">
          <textarea
            className="form-control"
            rows="10"
            value={card}
            onChange={e => setCards(e.target.value)}
          />
          <Button block outline color="primary" className="mt-2" onClick={fetchSynergies}>Fetch</Button>
        </Col>
        <Col xs="6">
          <h4>Synergy With</h4>        
            {synergies.map((card, index) => (
              <Card key={`${card.name}-${index}`} className="mb-2">
                <CardBody>
                  <img src={card.image} alt={card.name} />
                  {' '}{index+1}. {card.name} - {Math.round(card.rating * 10000) / 100}%
                </CardBody>
              </Card>
            ))}
        </Col>
      </Row>
    </>
  );
}

export default SynergyPage;
