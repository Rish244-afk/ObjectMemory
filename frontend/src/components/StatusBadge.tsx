import React from 'react';

interface StatusBadgeProps {
  status: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  const norm = status ? status.toUpperCase() : 'UNKNOWN';

  let colorClasses = 'bg-[#181D29] text-[#94A3B8] border-[#232A3B]';
  let dotColor = 'bg-[#64748B]';

  switch (norm) {
    case 'CONFIRMED_PRESENT':
    case 'CLEARLY_VISIBLE':
    case 'SUCCESS':
    case 'HEALTHY':
    case 'UNCHANGED':
      colorClasses = 'bg-emerald-950/40 text-emerald-300 border-emerald-800/50';
      dotColor = 'bg-emerald-400';
      break;

    case 'CONFIRMED_MOVED':
    case 'RELOCATED':
    case 'MOVED':
      colorClasses = 'bg-cyan-950/40 text-cyan-300 border-cyan-800/50';
      dotColor = 'bg-cyan-400';
      break;

    case 'CONFIRMED_ADDED':
    case 'NEWLY_APPEARED':
    case 'ADDED':
      colorClasses = 'bg-sky-950/40 text-sky-300 border-sky-800/50';
      dotColor = 'bg-sky-400';
      break;

    case 'PENDING_REMOVAL':
    case 'PARTIALLY_OCCLUDED':
    case 'MISSING_IN_QUERY':
    case 'DEGRADED':
      colorClasses = 'bg-amber-950/40 text-amber-300 border-amber-800/50';
      dotColor = 'bg-amber-400';
      break;

    case 'CONFIRMED_REMOVED':
    case 'ABSENT':
    case 'REMOVED':
      colorClasses = 'bg-rose-950/40 text-rose-300 border-rose-800/50';
      dotColor = 'bg-rose-400';
      break;

    case 'INSUFFICIENT_EVIDENCE':
    case 'AMBIGUOUS_IDENTITY':
    case 'UNCERTAIN':
      colorClasses = 'bg-purple-950/40 text-purple-300 border-purple-800/50';
      dotColor = 'bg-purple-400';
      break;

    case 'UNOBSERVED':
    case 'OUT_OF_FRAME':
    default:
      colorClasses = 'bg-[#181D29] text-[#94A3B8] border-[#232A3B]';
      dotColor = 'bg-[#64748B]';
      break;
  }

  const padding = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-medium rounded-full border ${colorClasses} ${padding} font-mono`}
      title={norm}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
      <span className="tracking-tight">{norm.replace(/_/g, ' ')}</span>
    </span>
  );
};
