function ResultRow({ index, card, ratingFormat = 'percent' }) {
  const rating = ratingFormat === 'inverse'
    ? Math.round((1 - card.rating) * 10000) / 100
    : Math.round(card.rating * 10000) / 100;

  return (
    <div className="d-flex align-items-center border-bottom py-1">
      <div className="text-muted me-2" style={{ width: 28, textAlign: 'right' }}>{index + 1}.</div>
      {card.image ? (
        <img
          src={card.image}
          alt={card.name}
          style={{ height: 40, width: 'auto', marginRight: 8, borderRadius: 3 }}
        />
      ) : (
        <div style={{ width: 40, marginRight: 8 }} />
      )}
      <div className="flex-grow-1 text-truncate">{card.name}</div>
      <div className="text-muted ms-2" style={{ fontVariantNumeric: 'tabular-nums' }}>
        {rating}%
      </div>
    </div>
  );
}

export default ResultRow;
