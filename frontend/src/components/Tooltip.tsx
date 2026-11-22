import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';

interface TooltipProps {
  content: string;
  children?: React.ReactNode;
}

export const Tooltip: React.FC<TooltipProps> = ({ content, children }) => {
  const [visible, setVisible] = useState(false);

  return (
    <span
      className="relative inline-flex items-center group cursor-help"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
      onFocus={() => setVisible(true)}
      onBlur={() => setVisible(false)}
      tabIndex={0}
      aria-label={content}
    >
      {children || <HelpCircle className="w-3.5 h-3.5 text-neutral-500 hover:text-neutral-300 transition" />}

      {visible && (
        <span
          role="tooltip"
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 p-2 rounded-lg bg-neutral-950 border border-neutral-700 text-[11px] text-neutral-300 shadow-2xl z-50 pointer-events-none leading-relaxed text-center font-sans font-normal animate-in fade-in zoom-in-95 duration-100"
        >
          {content}
          <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-neutral-700" />
        </span>
      )}
    </span>
  );
};
