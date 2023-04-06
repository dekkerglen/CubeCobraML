import { useCallback, useState } from 'react';
import { Row, Col, Button } from 'reactstrap';
import CardItem from './CardItem';

function RecommendPage() {
  const [cards, setCards] = useState('');
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
    console.log(json2);

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
        </Col>
      <Col xs="6">
        <h4>Recommended Removes</h4>
      </Col>
    </Row>
    </>
  );
}

export default RecommendPage;
