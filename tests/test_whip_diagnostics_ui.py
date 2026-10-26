"""Read-only diagnostics: frames, units, evidence and playback, not task success."""
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtCore, QtWidgets
from training.ui.whip_diagnostics_data import WhipDiagnosticsData, recording_choices
from training.ui.whip_diagnostics import WhippingDiagnostics


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.widgets=[]
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rest=np.full(10,.1)
        (self.root/'configuration.json').write_text(json.dumps({'model':{'cable':{'marker_interval_lengths_m':self.rest.tolist()}},'task':{'continuous_whip':{'version':'fixture'}}}))
        self.time=np.arange(5)*.01
        self.positions=np.zeros((5,11,3));self.positions[:,:,0]=np.linspace(0,1,11)
        self.velocities=np.zeros_like(self.positions)

    def tearDown(self):
        # close() only hides a QWidget. Delete the signal-connected hierarchy
        # while QApplication is alive, rather than during interpreter teardown.
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.widgets.clear()
        self.temp.cleanup()

    def save(self,name='motion.npz',**kwargs):
        arrays=dict(time_s=self.time,cable_positions=self.positions,cable_velocities=self.velocities)
        arrays.update(kwargs);np.savez(self.root/name,**arrays)
        return WhipDiagnosticsData.load(self.root,name)

    def test_stationary_extended_is_not_qualified(self):
        d=self.save()
        np.testing.assert_allclose(d.reach,1)
        np.testing.assert_allclose(d.joint_turn_deg,0)
        np.testing.assert_array_equal(d.relative_cosine,0)
        self.assertEqual(d.record,{})
        self.assertEqual(d.phase(4),'No recorded fold phase')

    def test_translation_world_relative_and_height_are_distinct(self):
        self.positions[:,:,2]=2
        self.velocities[:,:,0]=3
        d=self.save()
        np.testing.assert_allclose(d.world_outward,3)
        np.testing.assert_allclose(d.relative_outward,0)
        np.testing.assert_allclose(d.tip_height,0)

    def test_output_uses_world_x_not_radial_heading(self):
        self.positions[:,:,1]=self.positions[:,:,0];self.positions[:,:,0]=0
        self.velocities[:,-1,1]=2
        d=self.save()
        np.testing.assert_allclose(d.reach,0)
        np.testing.assert_allclose(d.world_outward,0)
        np.testing.assert_allclose(d.relative_outward,0)
        page=WhippingDiagnostics();self.widgets.append(page);page.resize(1100,800);page.show()
        page.set_run(self.root,[{'phase':'evaluation','round':1,'replay':'motion.npz'}])
        self.app.processEvents()

    def test_terminal_padding_is_not_displayed(self):
        d=self.save(terminal_time_s=.02)
        self.assertEqual(len(d.time),3)
        self.assertTrue(d.notices[0].startswith('Displaying 3/5'))

    def test_invalid_suffix_does_not_become_motion(self):
        self.positions[3:]=np.nan
        d=self.save(state_valid=np.array([1,1,1,0,0],bool),task_valid=False)
        self.assertEqual(len(d.time),3);self.assertFalse(d.validity)

    def test_fold_phase_comes_only_from_saved_log(self):
        record={'terminal':'event_end','qualified':False,'selected':None,'events':[
            {'kind':'armed','time_s':.02},{'kind':'opening_confirmed','time_s':.04}]}
        d=self.save(fold_event_json=json.dumps(record))
        self.assertEqual(d.phase(0),'No recorded fold milestone yet')
        self.assertEqual(d.phase(2),'Fold armed');self.assertEqual(d.phase(4),'Opening recorded')
        self.assertFalse(d.record['qualified'])

    def test_cannot_load_external_path_or_batch(self):
        with self.assertRaises(ValueError):WhipDiagnosticsData.load(self.root,'../motion.npz')
        with self.assertRaises(ValueError):self.save(cable_positions=np.zeros((2,5,11,3)))

    def test_choices_keep_failed_training_samples(self):
        (self.root/'collection-00004-motions.json').write_text(json.dumps({'samples':[
            {'lane':2,'replay':'lane2.npz','valid':False}]}))
        choices=recording_choices(self.root,[{'phase':'evaluation','round':4,'replay':'motion.npz'}])
        self.assertEqual(len(choices),2);self.assertIn('invalid',choices[1][0])

    def continuous_fixture(self):
        snapshot=json.loads((self.root/'configuration.json').read_text())
        snapshot['task']['continuous_whip']={'version':'continuous_directional_whip_four_second_v6'}
        (self.root/'configuration.json').write_text(json.dumps(snapshot))
        fields=np.array(['fold_support','straightening_mps','reward_rate','accrued_credit'])
        values=np.array([[0.,0.,0.,0.], [.2,.5,1.,.01], [.4,.8,2.,.03],
                         [.5,1.25,3.,.06], [np.nan,np.nan,np.nan,np.nan]])
        record=dict(contract='continuous_directional_whip_four_second_v6',terminal='workspace_failure',
                    terminal_time_s=.04,qualified=None,selected=None,
                    events=[dict(kind='workspace_failure',time_s=.04)])
        return dict(continuous_whip_fields=fields,continuous_whip_diagnostics=values,
                    fold_event_json=json.dumps(record),task_valid=False)

    def test_saved_continuous_fields_follow_prefix_trimming(self):
        fixture=self.continuous_fixture()
        d=self.save(**fixture,terminal_time_s=.02)
        self.assertEqual(d.mode,'continuous_directional_whip_four_second_v6')
        self.assertEqual(len(d.time),3)
        self.assertEqual(set(d.continuous),set(fixture['continuous_whip_fields']))
        for i,field in enumerate(fixture['continuous_whip_fields']):
            np.testing.assert_array_equal(d.continuous[field],fixture['continuous_whip_diagnostics'][:3,i])
        self.assertFalse(d.validity)
        self.assertIsNone(d.record['qualified'])

    def test_continuous_ui_reports_saved_values_and_missing_measurement(self):
        self.save(**self.continuous_fixture())
        page=WhippingDiagnostics();self.widgets.append(page);page.resize(1100,800);page.show()
        self.app.processEvents()
        page.set_run(self.root,[{'phase':'evaluation','round':1,'replay':'motion.npz'}])
        self.app.processEvents();page.seek(3)
        self.assertIn('continuous_directional_whip_four_second_v6',page.caption.text())
        self.assertIn('Saved continuous reward trace',page.telemetry.text())
        self.assertIn('Fold support: +0.50',page.telemetry.text())
        self.assertIn('Endpoint extension (m/s): +1.25',page.telemetry.text())
        self.assertIn('Reward rate: +3.00',page.telemetry.text())
        self.assertIn('Accrued credit: +0.06',page.telemetry.text())
        self.assertIn('INVALID',page.outcome.text())
        self.assertIn('Saved task qualification: not recorded',page.outcome.text())
        page.seek(4)
        self.assertIn('Reward rate: undefined',page.telemetry.text())
        self.assertNotIn('Reward rate: +0.00',page.telemetry.text())

    def test_tab_switch_seek_and_clear_do_not_reuse_old_data(self):
        self.save(task_valid=True)
        page=WhippingDiagnostics();self.widgets.append(page);page.resize(1100,700);page.show()
        self.app.processEvents()
        page.set_run(self.root,[{'phase':'evaluation','round':4,'replay':'motion.npz'}])
        self.app.processEvents();self.assertIsNotNone(page.data)
        page.slider.setValue(3);self.assertEqual(page.index,3)
        page.toggle();self.assertTrue(page.playing)
        page.hide();self.assertFalse(page.playing)
        page.show();self.app.processEvents()
        page.set_run(self.root,[]);self.assertIsNone(page.data)
        self.assertIsNone(page.canvas.data);self.assertEqual(page.timeline.rowCount(),0)
        page.close()


if __name__=='__main__':unittest.main()
