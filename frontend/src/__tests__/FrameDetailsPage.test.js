import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import FrameDetailsPage from '../FrameDetailsPage';

const sampleFrames = [
  {
    frame_index: 0,
    timestamp: 0.0,
    preview_path: '/static/frames/sample.jpg',
    retrieved_policies: [
      { policy_info: { category: 'Danger', title: 'Dangerous acts', description: 'Shows dangerous activity', source: 'db' }, similarity: 0.92 }
    ],
    policy: { description: 'Content showing dangerous or harmful activities', action_required: 'age_restriction_or_removal', severity: 'life-threatening' }
  }
];

test('clicking a retrieved policy opens the policy dialog', async () => {
  render(<FrameDetailsPage frames={sampleFrames} />);
  // find the list item by policy category text
  const item = await screen.findByText(/Dangerous acts/i);
  expect(item).toBeInTheDocument();
  // click the list item
  fireEvent.click(item);
  // dialog should open with title
  const dialogTitle = await screen.findByText(/Dangerous acts/i);
  expect(dialogTitle).toBeInTheDocument();
  // dialog should contain description
  const desc = await screen.findByText(/Shows dangerous activity/i);
  expect(desc).toBeInTheDocument();
});
