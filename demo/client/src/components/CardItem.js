import { Card, CardBody } from 'reactstrap';

function CardItem({ card }) {
  return (
    <Card>
      <img src={card.image} alt={card.name} />
      {card.name}
    </Card>
  );
}

export default CardItem;
