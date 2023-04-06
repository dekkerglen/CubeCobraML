import { useCallback, useState } from 'react';
import { Row, Col, Button, Card, CardBody } from 'reactstrap';
import CardItem from './CardItem';


function DraftPage() {
  const [cards, setCards] = useState('');
  const [pack, setPack] = useState([]);
  const [pool, setPool] = useState([]);
  const [picks, setPicks] = useState([]);

  const fetchPicks = useCallback(async () => {
    const response = await fetch('/api/cards', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cards: cards.split('\n') }),
    });
    const json = await response.json();    

    setPack(json.cards);
    
    const response2 = await fetch('/api/cards', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cards: cards.split('\n') }),
    });
    const json2 = await response.json();    

    setPack(json2.cards);

    const response3 = await fetch('/api/deckbuild', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cards: json.cards.map(card => card.name) }),
    });

    const json3 = await response3.json();

    setPicks(json3.picks);
  }, [cards]);
  

  return (
    <>
    <Row>
      <Col xs="6">
    <h3>Pack</h3>
        <textarea
          className="form-control"
          rows="10"
          value={cards}
          onChange={e => setCards(e.target.value)}
        />
      </Col>      
      <Col xs="6">
    <h3>Pack</h3>
        <textarea
          className="form-control"
          rows="10"
          value={cards}
          onChange={e => setCards(e.target.value)}
        />
      </Col>
      <Button block outline color="primary" className="mt-2" onClick={fetchPicks}>Fetch</Button>
  
    </Row>
    </>
  );
}

export default DraftPage;
