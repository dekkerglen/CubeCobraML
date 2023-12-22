import { useCallback, useState } from 'react';
import { Row, Col, Button, Card, CardBody } from 'reactstrap';
import CardItem from './CardItem';
import useLocalStorage from './hooks/useLocalStorage';

function DraftPage() {
  const [packText, setPackText] = useLocalStorage('packinput', '');
  const [poolText, setPoolText] = useLocalStorage('poolinput', '');
  const [pack, setPack] = useState([]);
  const [pool, setPool] = useState([]);
  const [picks, setPicks] = useState([]);

  const fetchPicks = useCallback(async () => {
    const response = await fetch('/api/cards', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cards: packText.split('\n') }),
    });
    const json = await response.json();    

    setPack(json.cards);
    
    const response2 = await fetch('/api/cards', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cards: poolText.split('\n') }),
    });
    const json2 = await response2.json();    

    setPool(json2.cards);

    const response3 = await fetch('/api/draft', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        pack: json.cards.map(card => card.name),
        pool: json2.cards.map(card => card.name) 
      }),
    });

    const json3 = await response3.json();


    setPicks(json3.picks);
  }, [packText, poolText]);
  

  return (
    <>
      <Row>
        <Col xs="6">
          <h3>Pack</h3>
          <textarea
            className="form-control"
            rows="10"
            value={packText}
            onChange={e => setPackText(e.target.value)}
          />
        </Col>      
        <Col xs="6">
          <h3>Pool</h3>
          <textarea
            className="form-control"
            rows="10"
            value={poolText}
            onChange={e => setPoolText(e.target.value)}
          />
        </Col>
        <Button block outline color="primary" className="my-2" onClick={fetchPicks}>Fetch</Button>
        <Col style={{ maxHeight:400, overflowY:'scroll' }} xs="6">
          <Row>
            {pack.map((card, index) => (
              <Col key={`${card.name}-${index}`} xs="3">
                <CardItem card={card} />
              </Col>
            ))}
          </Row>
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
        <Col className="mt-2" xs="12">
        <h4>Recommended Picks</h4>
          <Row>
            {picks.map((card, index) => (
            <Card key={`${card.name}-${index}`} className="mb-2">
              <CardBody>
                <img src={card.image} alt={card.name} />
                {' '}{index+1}. {card.name} - {Math.round(card.rating * 10000) / 100}%
              </CardBody>
            </Card>
            ))}
          </Row>
        </Col>
      </Row>
    </>
  );
}

export default DraftPage;
