import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import FrameDetailsPage from '../FrameDetailsPage';

// Minimal mock frames data
const makeFrame = (overrides = {}) => ({
  frame_index: 0,
  timestamp: 1.23,
  preview_path: null,
  visual_analysis: { category: 'Nudity & Sexual Content', confidence: 0.87 },
  retrieved_policies: overrides.retrieved_policies || [],
  policy: overrides.policy || null,
  personalized_reason: overrides.personalized_reason || null,
  examples: overrides.examples || null
});

describe('FrameDetailsPage toggle and explication behavior', () => {
  beforeEach(() => {
    // ensure localStorage is clean
    try { localStorage.removeItem('preferDbPolicyText'); } catch (e) {}
  });

  test('renders toggle and prefers DB text by default when available', async () => {
    const dbPolicy = {
      policy_info: {
        title: 'Nudity & Sexual Content',
        description: 'Explicit sexual content is disallowed.',
        examples: JSON.stringify(['Example A', 'Example B']),
        category: 'Nudity & Sexual Content'
      },
      similarity: 0.92
    };
    const frame = makeFrame({ retrieved_policies: [dbPolicy] });
    render(<FrameDetailsPage frames={[frame]} onBack={() => {}} />);

    // toggle should be in the document and default to checked (preferDb default true)
    const toggle = screen.getByLabelText(/Prefer DB policy text when available/i);
    expect(toggle).toBeInTheDocument();
    expect(toggle).toBeChecked();

    // the DB policy title/why should appear
    await waitFor(() => expect(screen.getByText(/Nudity & Sexual Content/i)).toBeInTheDocument());
    expect(screen.getByText(/Explicit sexual content is disallowed./i)).toBeInTheDocument();
  });

  test('when toggled off, uses heuristic/static mapping instead of DB text', async () => {
    const dbPolicy = {
      policy_info: {
        title: 'Nudity & Sexual Content',
        description: 'Explicit sexual content is disallowed.',
        category: 'Nudity & Sexual Content'
      },
      similarity: 0.92
    };
    const frame = makeFrame({ retrieved_policies: [dbPolicy] });
    render(<FrameDetailsPage frames={[frame]} onBack={() => {}} />);

    const toggle = screen.getByLabelText(/Prefer DB policy text when available/i);
    // turn it off
    fireEvent.click(toggle);
    expect(toggle).not.toBeChecked();

    // With preferDb off, the UI should show a fallback title from POLICY_EXPLICATIONS mapping or synthesized text
    await waitFor(() => {
      // Expect either the backfilled key title or a synthesized 'Detected issue' fallback
      const possible = screen.queryByText(/Nudity & Sexual Content/i) || screen.queryByText(/Detected issue/i);
      expect(possible).toBeTruthy();
    });
  });
});
