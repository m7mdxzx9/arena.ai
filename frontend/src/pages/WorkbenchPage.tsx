import Workbench from '../labs/Workbench'
import { useGame } from '../ui'

export default function WorkbenchPage({ dataset }: { dataset?: string }) {
  const { ov } = useGame()
  return (
    <div className="stack">
      <div className="topbar">
        <div><div className="kicker">ML Workshop · Workbench</div><h1 style={{ margin: 0 }}>ML Workbench</h1></div>
      </div>
      {ov.player.mode <= 2 && (
        <div className="info-box">
          <b>How to use the Workbench:</b> pick a dataset → choose which columns the model may look at (features) → pick an algorithm → train. You'll get real
          test-set metrics, a confusion matrix, feature influence and a diagnosis. Every run is saved to <a href="#/history">Experiment History</a> so you can compare models fairly.
        </div>
      )}
      <Workbench key={dataset || 'default'} dataset={dataset} context="workbench" />
    </div>
  )
}
