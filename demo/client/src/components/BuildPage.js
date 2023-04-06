import { useCallback, useState } from 'react';
import { Row, Col, Button, Card, CardBody } from 'reactstrap';
import CardItem from './CardItem';

function BuildPage() {
  const [cards, setCards] = useState('');
  const [pool, setPool] = useState([]);
  const [mainboard, setMainboard] = useState([]);
  const [sideboard, setSideboard] = useState([]);

  const fetchPools = useCallback(async () => {
    const response = await fetch('/api/cards', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cards: cards.split('\n') }),
    });
    const json = await response.json();    

    setPool(json.cards);

    const response2 = await fetch('/api/deckbuild', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cards: json.cards.map(card => card.name) }),
    });

    const json2 = await response2.json();

    setMainboard(json2.mainboard);
    setSideboard(json2.sideboard);
  }, [cards]);
  

  return (
    <>
    <h3>Pool</h3>
    <Row>
      <Col xs="6">
        <textarea
          className="form-control"
          rows="10"
          value={cards}
          onChange={e => setCards(e.target.value)}
        />
        <Button block outline color="primary" className="mt-2" onClick={fetchPools}>Fetch</Button>
      </Col>
      <Col style={{ maxHeight:400, overflowY:'scroll' }} xs="6">
        <Row>
          {pool.map((card, index) => (
            <Col key={`${card.name}-${index}`} xs="3">
              <CardItem card={card} />
            </Col>
          ))}
        </Row>
      </Col>
      <Col xs="6">
        <h4>Mainboard</h4>        
          {mainboard.map((card, index) => (
            <Card key={`${card.name}-${index}`} className="mb-2">
              <CardBody>
                <img src={card.image} alt={card.name} />
                {' '}{index+1}. {card.name} - {Math.round(card.rating * 10000) / 100}%
              </CardBody>
            </Card>
          ))}
      </Col>
      <Col xs="6">
        <h4>Sideboard</h4>
          {sideboard.map((card, index) => (
            <Card key={`${card.name}-${index}`} className="mb-2">
              <CardBody>
                <img src={card.image} alt={card.name} />
                {' '}{index+1}. {card.name} - {Math.round((1-card.rating) * 10000) / 100}%
              </CardBody>
            </Card>
          ))}
      </Col>
    </Row>
    </>
  );
}

export default BuildPage;
