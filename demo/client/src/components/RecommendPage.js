import { useCallback, useState } from 'react';
import { Row, Col, Button, Card, CardBody } from 'reactstrap';
import CardItem from './CardItem';
import useLocalStorage from './hooks/useLocalStorage';

function RecommendPage() {
  const [cards, setCards] = useLocalStorage('recommendinput', '');
  const [cube, setCube] = useState([]);
  const [adds, setAdds] = useState([]);
  const [removes, setRemoves] = useState([]);

  const fetchCubes = useCallback(async () => {
    const response = await fetch('/api/cards', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cards: cards.split('\n') }),
    });
    const json = await response.json();    

    setCube(json.cards);

    const response2 = await fetch('/api/recommend', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ cards: json.cards.map(card => card.name) }),
    });

    const json2 = await response2.json();

    setAdds(json2.adds);
    setRemoves(json2.removes);
  }, [cards]);
  

  return (
    <>
    <h3>Recommend</h3>
    <Row>
      <Col xs="6">
        <textarea
          className="form-control"
          rows="10"
          value={cards}
          onChange={e => setCards(e.target.value)}
        />
        <Button block outline color="primary" className="mt-2" onClick={fetchCubes}>Fetch</Button>
      </Col>
      <Col style={{ maxHeight:400, overflowY:'scroll' }} xs="6">
        <Row>
          {cube.map((card, index) => (
            <Col key={`${card.name}-${index}`} xs="3">
              <CardItem card={card} />
            </Col>
          ))}
        </Row>
      </Col>
      <Col xs="6">
        <h4>Recommended Adds</h4>        
          {adds.map((card, index) => (
            <Card key={`${card.name}-${index}`} className="mb-2">
              <CardBody>
                <img src={card.image} alt={card.name} />
                {' '}{index+1}. {card.name} - {Math.round(card.rating * 10000) / 100}%
              </CardBody>
            </Card>
          ))}
      </Col>
      <Col xs="6">
        <h4>Recommended Removes</h4>
          {removes.map((card, index) => (
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

export default RecommendPage;
